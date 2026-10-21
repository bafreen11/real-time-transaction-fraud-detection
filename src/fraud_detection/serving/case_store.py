"""SQLite store for scored transactions, alert cases and analyst feedback.

Every transaction the scoring service scores is saved with the exact
feature vector the model saw. Transactions flagged FRAUD_ALERT open a
case that analysts move through CASE_STATUSES and label with feedback.
Labeled rows are fed back into retraining, and recent rows are the
"current" data for drift monitoring.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone

import pandas as pd

CASE_STATUSES = [
    "OPEN",
    "INVESTIGATING",
    "CONFIRMED_FRAUD",
    "FALSE_POSITIVE",
    "CLOSED",
]
FEEDBACK_LABELS = {"FRAUD": 1, "LEGITIMATE": 0}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scored_transactions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id    INTEGER,
    scored_at         TEXT NOT NULL,
    step              INTEGER NOT NULL,
    type              TEXT,
    amount            REAL NOT NULL,
    name_orig         TEXT,
    name_dest         TEXT,
    transaction_json  TEXT NOT NULL,
    features_json     TEXT NOT NULL,
    xgboost_probability REAL NOT NULL,
    gnn_probability   REAL NOT NULL,
    fraud_score       REAL NOT NULL,
    threshold         REAL NOT NULL,
    decision          TEXT NOT NULL,
    model_version     TEXT,
    actual_is_fraud   INTEGER,
    case_status       TEXT,
    analyst_label     TEXT,
    analyst_note      TEXT,
    updated_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_scored_decision ON scored_transactions (decision);
CREATE INDEX IF NOT EXISTS idx_scored_status ON scored_transactions (case_status);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row):
    record = dict(row)
    record["transaction"] = json.loads(record.pop("transaction_json"))
    record["features"] = json.loads(record.pop("features_json"))
    return record


class CaseStore:
    def __init__(self, db_path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def add_scored(self, transaction, features, result, actual_is_fraud=None):
        is_alert = result["decision"] == "FRAUD_ALERT"
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO scored_transactions (
                    transaction_id, scored_at, step, type, amount,
                    name_orig, name_dest, transaction_json, features_json,
                    xgboost_probability, gnn_probability, fraud_score,
                    threshold, decision, model_version, actual_is_fraud,
                    case_status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction.get("transaction_id"),
                    _now(),
                    int(transaction["step"]),
                    transaction.get("type"),
                    float(transaction["amount"]),
                    transaction.get("nameOrig"),
                    transaction.get("nameDest"),
                    json.dumps(transaction),
                    json.dumps(features),
                    result["xgboost_probability"],
                    result["gnn_probability"],
                    result["fraud_score"],
                    result["threshold"],
                    result["decision"],
                    result.get("model_version"),
                    actual_is_fraud,
                    "OPEN" if is_alert else None,
                    _now(),
                ),
            )
            return cursor.lastrowid

    def list_transactions(self, limit=100, alerts_only=False, status=None):
        query = "SELECT * FROM scored_transactions"
        conditions, params = [], []
        if alerts_only:
            conditions.append("decision = 'FRAUD_ALERT'")
        if status:
            conditions.append("case_status = ?")
            params.append(status)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get(self, record_id):
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM scored_transactions WHERE id = ?", (record_id,)
            ).fetchone()
        return _row_to_dict(row) if row else None

    def update_case(self, record_id, status=None, label=None, note=None):
        if status is not None and status not in CASE_STATUSES:
            raise ValueError(f"Unknown case status: {status}")
        if label is not None and label not in FEEDBACK_LABELS:
            raise ValueError(f"Unknown feedback label: {label}")

        updates, params = ["updated_at = ?"], [_now()]
        for column, value in (
            ("case_status", status),
            ("analyst_label", label),
            ("analyst_note", note),
        ):
            if value is not None:
                updates.append(f"{column} = ?")
                params.append(value)
        params.append(record_id)

        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE scored_transactions SET {', '.join(updates)} "
                "WHERE id = ?",
                params,
            )
        return self.get(record_id)

    def stats(self):
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*) AS scored,
                    SUM(decision = 'FRAUD_ALERT') AS alerts,
                    SUM(case_status IN ('OPEN', 'INVESTIGATING')) AS open_cases,
                    SUM(analyst_label = 'FRAUD') AS labeled_fraud,
                    SUM(analyst_label = 'LEGITIMATE') AS labeled_legitimate,
                    SUM(decision = 'FRAUD_ALERT' AND actual_is_fraud = 1)
                        AS true_alerts,
                    SUM(actual_is_fraud = 1) AS actual_fraud,
                    MAX(step) AS latest_step
                FROM scored_transactions
                """
            ).fetchone()
        return {key: (row[key] or 0) for key in row.keys()}

    def recent_features(self, limit):
        """Feature vectors of the most recent scored transactions."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT features_json FROM scored_transactions "
                "ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return pd.DataFrame([json.loads(r["features_json"]) for r in rows])

    def labeled_features(self):
        """Feature vectors and 0/1 labels of analyst-labeled rows."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT features_json, analyst_label FROM scored_transactions "
                "WHERE analyst_label IS NOT NULL"
            ).fetchall()
        if not rows:
            return pd.DataFrame(), pd.Series(dtype=int)
        features = pd.DataFrame([json.loads(r["features_json"]) for r in rows])
        labels = pd.Series(
            [FEEDBACK_LABELS[r["analyst_label"]] for r in rows], name="isFraud"
        )
        return features, labels
