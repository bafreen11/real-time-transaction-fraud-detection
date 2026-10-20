import json
import time
import pandas as pd
from kafka import KafkaProducer

from fraud_detection.config import (
    FEATURE_COLUMNS_PATH as FEATURE_PATH,
    FEATURES_DATA_PATH as DATA_PATH,
)

TOPIC = "transactions"
BOOTSTRAP_SERVERS = "localhost:9092"
BATCH_SIZE = 10

# Load the exact feature order used by the trained model
with open(FEATURE_PATH, "r") as file:
    feature_columns = json.load(file)

# Read only a small batch for our first test
df = pd.read_csv(DATA_PATH, nrows=BATCH_SIZE)

# Verify that all model features exist in the CSV
missing = [col for col in feature_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing model features: {missing}")

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda data: json.dumps(data).encode("utf-8")
)

try:
    for index, row in df.iterrows():

        # Features that the XGBoost model expects
        features = {
            col: (
                None if pd.isna(row[col])
                else row[col].item()
                if hasattr(row[col], "item")
                else row[col]
            )
            for col in feature_columns
        }

        # Metadata is separate from the model features
        message = {
            "transaction_id": int(index + 1),
            "features": features,
            "transaction_type": str(row["type"]),
            "nameOrig": str(row["nameOrig"]),
            "nameDest": str(row["nameDest"]),
            "actual_is_fraud": int(row["isFraud"])
        }

        # Publish transaction to Kafka
        future = producer.send(TOPIC, value=message)
        metadata = future.get(timeout=10)

        print(
            f"Sent transaction {index + 1} | "
            f"Type: {row['type']} | "
            f"Topic: {metadata.topic} | "
            f"Partition: {metadata.partition}"
        )

        time.sleep(0.2)

    producer.flush()
    print(f"\nSuccessfully streamed {len(df)} transactions to Kafka.")

finally:
    producer.close()
