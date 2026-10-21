"""Evidently data-drift check with an automatic retraining trigger.

Compares the feature vectors of the most recent scored transactions
(the scoring service's case store) with the drift reference saved by
the stream model training. When the share of drifted features reaches
DRIFT_SHARE_THRESHOLD, it retrains (scripts/feast/train_stream_model.py),
unless a retrain ran within RETRAIN_COOLDOWN_HOURS, and asks the running
scoring service to reload the model if a new one was promoted.

Environment variables:
    DRIFT_CHECK_INTERVAL_SECONDS  repeat forever with this pause (default:
                                  run once, e.g. from a Kubernetes CronJob)
    RETRAIN_ON_DRIFT              0 = report only (default 1)
    FRAUD_SERVICE_URL             service to reload after a retrain
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import mlflow
import pandas as pd
import requests
from evidently import Report
from evidently.presets import DataDriftPreset

from fraud_detection.config import (
    DRIFT_LATEST_REPORT_PATH,
    DRIFT_LATEST_SUMMARY_PATH,
    DRIFT_MIN_ROWS,
    DRIFT_REFERENCE_PATH,
    DRIFT_REPORTS_DIR,
    DRIFT_SHARE_THRESHOLD,
    DRIFT_WINDOW_SIZE,
    MLFLOW_DRIFT_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
    RETRAIN_COOLDOWN_HOURS,
    RETRAIN_STATE_PATH,
    SERVING_DB_PATH,
    ensure_output_dirs,
)
from fraud_detection.serving.case_store import CaseStore

INTERVAL_SECONDS = int(os.environ.get("DRIFT_CHECK_INTERVAL_SECONDS", "0"))
RETRAIN_ON_DRIFT = os.environ.get("RETRAIN_ON_DRIFT", "1") == "1"
FRAUD_SERVICE_URL = os.environ.get("FRAUD_SERVICE_URL", "http://127.0.0.1:8001")
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "feast" / "train_stream_model.py"

ensure_output_dirs()


def column_drifts(snapshot):
    """Per-feature drift flags from an Evidently snapshot."""
    drifts = {}
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if not config["type"].endswith("ValueDrift"):
            continue
        value = float(metric["value"])
        # p-value tests drift below the threshold, distances above it.
        if "p_value" in config["method"]:
            drifted = value < config["threshold"]
        else:
            drifted = value >= config["threshold"]
        drifts[config["column"]] = {
            "method": config["method"],
            "value": value,
            "drifted": bool(drifted),
        }
    return drifts


def retrain_allowed():
    if not RETRAIN_STATE_PATH.exists():
        return True
    last = datetime.fromisoformat(
        json.loads(RETRAIN_STATE_PATH.read_text())["last_retrain"]
    )
    return datetime.now(timezone.utc) - last >= timedelta(
        hours=RETRAIN_COOLDOWN_HOURS
    )


def retrain(reason):
    print("Triggering retraining:", reason)
    RETRAIN_STATE_PATH.write_text(json.dumps({
        "last_retrain": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }))
    completed = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=PROJECT_ROOT,
        env={**os.environ, "RETRAIN_REASON": reason},
    )
    if completed.returncode != 0:
        print("Retraining failed with exit code", completed.returncode)
        return False

    try:
        response = requests.post(f"{FRAUD_SERVICE_URL}/model/reload", timeout=30)
        print("Scoring service reloaded:", response.json())
    except requests.RequestException as exc:
        print("Could not reach the scoring service to reload:", exc)
    return True


def check_once():
    checked_at = datetime.now(timezone.utc)
    summary = {"checked_at": checked_at.isoformat(timespec="seconds")}

    if not DRIFT_REFERENCE_PATH.exists() or not SERVING_DB_PATH.exists():
        summary["status"] = "waiting: no reference data or no scored transactions"
        DRIFT_LATEST_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        print(summary["status"])
        return summary

    reference = pd.read_parquet(DRIFT_REFERENCE_PATH)
    current = CaseStore(SERVING_DB_PATH).recent_features(DRIFT_WINDOW_SIZE)

    summary["current_rows"] = len(current)
    if len(current) < DRIFT_MIN_ROWS:
        summary["status"] = (
            f"waiting: {len(current)} scored transactions, "
            f"need {DRIFT_MIN_ROWS}"
        )
        DRIFT_LATEST_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        print(summary["status"])
        return summary

    current = current[reference.columns]

    snapshot = Report([DataDriftPreset()]).run(
        current_data=current, reference_data=reference
    )
    drifts = column_drifts(snapshot)
    drifted = sorted(c for c, d in drifts.items() if d["drifted"])
    share = len(drifted) / max(len(drifts), 1)

    stamp = checked_at.strftime("%Y%m%d_%H%M%S")
    report_path = DRIFT_REPORTS_DIR / f"drift_report_{stamp}.html"
    snapshot.save_html(str(report_path))
    snapshot.save_html(str(DRIFT_LATEST_REPORT_PATH))

    drift_detected = share >= DRIFT_SHARE_THRESHOLD
    summary.update({
        "status": "drift detected" if drift_detected else "no significant drift",
        "reference_rows": len(reference),
        "drift_share": round(share, 4),
        "drift_share_threshold": DRIFT_SHARE_THRESHOLD,
        "drifted_columns": drifted,
        "columns": drifts,
        "report": report_path.name,
        "retrain_triggered": False,
    })

    if drift_detected and RETRAIN_ON_DRIFT:
        if retrain_allowed():
            summary["retrain_triggered"] = True
            summary["retrain_succeeded"] = retrain(
                f"drift share {share:.2f} >= {DRIFT_SHARE_THRESHOLD} "
                f"({', '.join(drifted)})"
            )
        else:
            summary["retrain_skipped"] = (
                f"cooldown of {RETRAIN_COOLDOWN_HOURS}h not over"
            )

    DRIFT_LATEST_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_DRIFT_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"drift_check_{stamp}"):
        mlflow.log_metrics({
            "drift_share": share,
            "drifted_columns": len(drifted),
            "current_rows": len(current),
        })
        mlflow.set_tags({
            "drift_detected": str(drift_detected),
            "retrain_triggered": str(summary["retrain_triggered"]),
        })
        mlflow.log_artifact(str(report_path))

    print(f"Drift share {share:.2f} | drifted: {drifted or 'none'} | "
          f"retrain triggered: {summary['retrain_triggered']}")
    return summary


check_once()

while INTERVAL_SECONDS > 0:
    time.sleep(INTERVAL_SECONDS)
    try:
        check_once()
    except Exception as exc:
        print("Drift check failed:", exc)
