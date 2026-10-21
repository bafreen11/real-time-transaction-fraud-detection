import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from fraud_detection.config import (
    DRIFT_REFERENCE_PATH,
    GNN_SERVING_ARTIFACT_PATH,
    MLFLOW_STREAM_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    RAW_DATA_PATH,
    RETRAIN_PROMOTION_TOLERANCE,
    SERVING_DB_PATH,
    SHARED_TRAIN_PATH,
    STREAM_BLEND_GNN_WEIGHTS,
    STREAM_EVAL_FRACTION,
    STREAM_MODEL_ARCHIVE_DIR,
    STREAM_MODEL_REGISTRY_NAME,
    STREAM_THRESHOLDS,
    STREAM_XGBOOST_MODEL_PATH,
    ensure_output_dirs,
)
from fraud_detection.data.splits import shared_split
from fraud_detection.feature_store import get_historical_state, get_store
from fraud_detection.features.behavioral import (
    RECIPIENT_STATE_FIELDS,
    add_event_timestamps,
    is_tracked_recipient,
)
from fraud_detection.features.stream import (
    STREAM_FEATURE_COLUMNS,
    build_stream_features,
)
from fraud_detection.models.gnn_serving import GNNServingScorer
from fraud_detection.serving.case_store import CaseStore
from fraud_detection.serving.scorer import blend_scores

# Set by the drift monitor when it triggers a retrain.
RETRAIN_REASON = os.environ.get("RETRAIN_REASON", "manual")
FEAST_CHUNK_SIZE = 150_000
DRIFT_REFERENCE_ROWS = 20_000

ensure_output_dirs()

# --------------------------------------------------
# 1. Load raw data and recreate the shared split
# --------------------------------------------------

print("Loading raw PaySim transactions...")
df = pd.read_csv(
    RAW_DATA_PATH,
    usecols=[
        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
        "newbalanceOrig", "nameDest", "oldbalanceDest",
        "newbalanceDest", "isFraud",
    ],
    dtype={"type": "category"},
)
df = add_event_timestamps(df)

train_df, test_df = shared_split(df)
del df

# The split must be the one the shared GNN was trained on.
saved_train_amounts = pd.read_csv(SHARED_TRAIN_PATH, usecols=["amount"])
if not np.allclose(
    saved_train_amounts["amount"].to_numpy(), train_df["amount"].to_numpy()
):
    raise ValueError("Recreated split does not match shared_train.csv.")

# A stratified sample of the held-out test partition, halved into a
# validation set (threshold and blend weight) and a final test set.
eval_df, _ = train_test_split(
    test_df,
    train_size=STREAM_EVAL_FRACTION,
    stratify=test_df["isFraud"],
    random_state=RANDOM_STATE,
)
val_df, final_test_df = train_test_split(
    eval_df,
    test_size=0.5,
    stratify=eval_df["isFraud"],
    random_state=RANDOM_STATE,
)
del test_df, eval_df

print("Training rows:", len(train_df), "| fraud:", int(train_df["isFraud"].sum()))
print("Validation rows:", len(val_df), "| fraud:", int(val_df["isFraud"].sum()))
print("Test rows:", len(final_test_df), "| fraud:", int(final_test_df["isFraud"].sum()))

# --------------------------------------------------
# 2. Point-in-time recipient state from Feast
# --------------------------------------------------

store = get_store()


def feast_state(rows):
    """Recipient state for each row as of just before the transaction."""
    state = pd.DataFrame(
        np.nan, index=range(len(rows)), columns=RECIPIENT_STATE_FIELDS
    )
    tracked = np.flatnonzero(is_tracked_recipient(rows["nameDest"]))

    for start in range(0, len(tracked), FEAST_CHUNK_SIZE):
        positions = tracked[start:start + FEAST_CHUNK_SIZE]
        chunk = rows.iloc[positions]
        state.iloc[positions] = get_historical_state(
            store, chunk["nameDest"], chunk["lookup_timestamp"]
        ).to_numpy(dtype=float)

    return state


def stream_features(rows):
    print(f"  Fetching Feast features for {len(rows)} rows...")
    return build_stream_features(rows, feast_state(rows))


print("\nBuilding features from Feast:")
X_train = stream_features(train_df)
X_val = stream_features(val_df)
X_test = stream_features(final_test_df)

y_train = train_df["isFraud"].to_numpy()
y_val = val_df["isFraud"].to_numpy()
y_test = final_test_df["isFraud"].to_numpy()

# --------------------------------------------------
# 3. Add analyst feedback labels from the scoring service
# --------------------------------------------------

feedback_rows = 0
if SERVING_DB_PATH.exists():
    feedback_X, feedback_y = CaseStore(SERVING_DB_PATH).labeled_features()
    if len(feedback_X):
        feedback_rows = len(feedback_X)
        X_train = pd.concat(
            [X_train, feedback_X[STREAM_FEATURE_COLUMNS]], ignore_index=True
        )
        y_train = np.concatenate([y_train, feedback_y.to_numpy()])

print("Analyst feedback rows added:", feedback_rows)

# --------------------------------------------------
# 4. Train XGBoost with class weighting
# --------------------------------------------------

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": float(scale_pos_weight),
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

print("\nTraining stream XGBoost model...")
model = XGBClassifier(**params)
model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Pick GNN blend weight and threshold on validation
# --------------------------------------------------

gnn = GNNServingScorer(joblib.load(GNN_SERVING_ARTIFACT_PATH))

val_xgb = model.predict_proba(X_val)[:, 1]
val_gnn = gnn.score(val_df)

selection = []
for gnn_weight in STREAM_BLEND_GNN_WEIGHTS:
    blended = blend_scores(val_xgb, val_gnn, gnn_weight)
    for threshold in STREAM_THRESHOLDS:
        predictions = (blended >= threshold).astype(int)
        selection.append({
            "gnn_weight": gnn_weight,
            "threshold": threshold,
            "average_precision": average_precision_score(y_val, blended),
            "precision": precision_score(y_val, predictions, zero_division=0),
            "recall": recall_score(y_val, predictions, zero_division=0),
            "f1": f1_score(y_val, predictions, zero_division=0),
        })

selection = pd.DataFrame(selection)
best = selection.sort_values(
    ["f1", "average_precision"], ascending=False
).iloc[0]
gnn_weight = float(best["gnn_weight"])
threshold = float(best["threshold"])

print(f"Selected on validation: GNN weight {gnn_weight}, "
      f"threshold {threshold}, F1 {best['f1']:.4f}")

# --------------------------------------------------
# 6. Final test evaluation (PR-AUC first, accuracy is not used)
# --------------------------------------------------


def evaluate(scores, cutoff):
    predictions = (scores >= cutoff).astype(int)
    return {
        "average_precision": average_precision_score(y_test, scores),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }


test_xgb = model.predict_proba(X_test)[:, 1]
test_gnn = gnn.score(final_test_df)
test_blend = blend_scores(test_xgb, test_gnn, gnn_weight)

results = {
    "blend": evaluate(test_blend, threshold),
    "xgboost_only": evaluate(test_xgb, threshold),
    "gnn_only": evaluate(test_gnn, 0.5),
}

print("\n========== STREAM MODEL TEST RESULTS ==========")
print(pd.DataFrame(results).T.round(4).to_string())

# --------------------------------------------------
# 7. Champion / challenger: compare with the current model
# --------------------------------------------------

current_ap = None
if STREAM_XGBOOST_MODEL_PATH.exists():
    current = joblib.load(STREAM_XGBOOST_MODEL_PATH)
    current_scores = blend_scores(
        current["model"].predict_proba(X_test[current["feature_columns"]])[:, 1],
        test_gnn,
        current["gnn_weight"],
    )
    current_ap = average_precision_score(y_test, current_scores)
    print(f"\nCurrent model ({current.get('version')}) test AP: {current_ap:.4f}")

new_ap = results["blend"]["average_precision"]
promote = (
    current_ap is None
    or new_ap >= current_ap - RETRAIN_PROMOTION_TOLERANCE
)

version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
artifact = {
    "model": model,
    "feature_columns": STREAM_FEATURE_COLUMNS,
    "threshold": threshold,
    "gnn_weight": gnn_weight,
    "version": version,
}

candidate_path = STREAM_MODEL_ARCHIVE_DIR / f"stream_xgboost_{version}.joblib"
joblib.dump(artifact, candidate_path)

if promote:
    if STREAM_XGBOOST_MODEL_PATH.exists():
        shutil.copy2(
            STREAM_XGBOOST_MODEL_PATH,
            STREAM_MODEL_ARCHIVE_DIR / "stream_xgboost_previous.joblib",
        )
    shutil.copy2(candidate_path, STREAM_XGBOOST_MODEL_PATH)
    print(f"Promoted model {version} -> {STREAM_XGBOOST_MODEL_PATH}")

    # Validation rows follow the natural class balance, so they are
    # the reference distribution for drift monitoring.
    X_val.sample(
        n=min(DRIFT_REFERENCE_ROWS, len(X_val)), random_state=RANDOM_STATE
    ).to_parquet(DRIFT_REFERENCE_PATH, index=False)
    print("Saved drift reference:", DRIFT_REFERENCE_PATH)
else:
    print(f"Not promoted: test AP {new_ap:.4f} is below the current "
          f"model's {current_ap:.4f}. Candidate kept at {candidate_path}")

# --------------------------------------------------
# 8. Track the run in MLflow
# --------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_STREAM_EXPERIMENT_NAME)

with mlflow.start_run(run_name=f"stream_xgboost_{version}"):
    mlflow.set_tags({
        "trigger": RETRAIN_REASON,
        "promoted": str(promote),
        "model_version": version,
    })
    mlflow.log_params({
        **{k: v for k, v in params.items() if k != "n_jobs"},
        "feature_count": len(STREAM_FEATURE_COLUMNS),
        "train_rows": len(X_train),
        "feedback_rows": feedback_rows,
        "eval_fraction": STREAM_EVAL_FRACTION,
        "gnn_weight": gnn_weight,
        "threshold": threshold,
    })
    for name, metrics in results.items():
        for metric, value in metrics.items():
            mlflow.log_metric(f"test_{name}_{metric}", float(value))
    if current_ap is not None:
        mlflow.log_metric("test_current_model_average_precision", current_ap)

    with tempfile.TemporaryDirectory() as tmp:
        selection_path = Path(tmp) / "validation_selection.csv"
        selection.to_csv(selection_path, index=False)
        mlflow.log_artifact(str(selection_path))
    mlflow.log_artifact(str(candidate_path))

    if promote:
        try:
            mlflow.xgboost.log_model(
                model,
                name="model",
                registered_model_name=STREAM_MODEL_REGISTRY_NAME,
            )
        except Exception as exc:
            print("Model registry step skipped:", exc)

    print("MLflow run:", mlflow.active_run().info.run_id)

print("Stream model training finished.")
