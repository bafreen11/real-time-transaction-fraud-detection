"""Replay PaySim transactions into Kafka in original file order.

Starts at REPLAY_START_STEP, the step the Feast online store was
bootstrapped up to (scripts/feast/build_feature_store.py), so the
scoring service sees recipient history exactly as training did.

Environment variables:
    REPLAY_START_STEP        first step to replay (config default)
    REPLAY_MAX_TRANSACTIONS  stop after this many (default 2000, 0 = all)
    REPLAY_DELAY_SECONDS     pause between messages (default 0.05)
    REPLAY_FRAUD_ONLY_TYPES  1 = only TRANSFER/CASH_OUT (default 0); the
                             skipped rows are then missing from Feast
"""

import json
import os
import time

import pandas as pd
from kafka import KafkaProducer

from fraud_detection.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TRANSACTIONS_TOPIC,
    RAW_DATA_PATH,
    REPLAY_START_STEP,
)
from fraud_detection.features.engineering import TRANSACTION_FIELDS

MAX_TRANSACTIONS = int(os.environ.get("REPLAY_MAX_TRANSACTIONS", "2000"))
DELAY_SECONDS = float(os.environ.get("REPLAY_DELAY_SECONDS", "0.05"))
FRAUD_ONLY_TYPES = os.environ.get("REPLAY_FRAUD_ONLY_TYPES", "0") == "1"
CHUNK_SIZE = 200_000

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda data: json.dumps(data).encode("utf-8"),
    linger_ms=20,
)

print(f"Replaying from step {REPLAY_START_STEP} to topic "
      f"{KAFKA_TRANSACTIONS_TOPIC} at {KAFKA_BOOTSTRAP_SERVERS}")

sent = 0
fraud_sent = 0
row_offset = 0

try:
    for chunk in pd.read_csv(RAW_DATA_PATH, chunksize=CHUNK_SIZE):
        # transaction_id is the row position in the raw file, the same
        # id add_event_timestamps() assigns.
        chunk.index = range(row_offset, row_offset + len(chunk))
        row_offset += len(chunk)

        chunk = chunk[chunk["step"] >= REPLAY_START_STEP]
        if FRAUD_ONLY_TYPES:
            chunk = chunk[chunk["type"].isin(["TRANSFER", "CASH_OUT"])]

        for transaction_id, row in chunk.iterrows():
            transaction = {
                "step": int(row["step"]),
                "type": str(row["type"]),
                "amount": float(row["amount"]),
                "nameOrig": str(row["nameOrig"]),
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "newbalanceOrig": float(row["newbalanceOrig"]),
                "nameDest": str(row["nameDest"]),
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "newbalanceDest": float(row["newbalanceDest"]),
            }

            producer.send(KAFKA_TRANSACTIONS_TOPIC, value={
                "transaction_id": int(transaction_id),
                "transaction": transaction,
                # Kept for streaming/kafka_fraud_consumer.py (baseline API)
                "features": {f: transaction[f] for f in TRANSACTION_FIELDS},
                "transaction_type": transaction["type"],
                "nameOrig": transaction["nameOrig"],
                "nameDest": transaction["nameDest"],
                "actual_is_fraud": int(row["isFraud"]),
            })

            sent += 1
            fraud_sent += int(row["isFraud"])

            if sent % 100 == 0:
                print(f"Sent {sent} transactions (step {transaction['step']}, "
                      f"{fraud_sent} actual fraud)")

            if MAX_TRANSACTIONS and sent >= MAX_TRANSACTIONS:
                raise StopIteration

            if DELAY_SECONDS:
                time.sleep(DELAY_SECONDS)

except StopIteration:
    pass

finally:
    producer.flush()
    producer.close()

print(f"\nReplay finished: {sent} transactions, {fraud_sent} actual fraud.")
