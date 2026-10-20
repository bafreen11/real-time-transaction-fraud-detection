
import json
import pandas as pd
import joblib

from kafka import KafkaConsumer

# Project paths
MODEL_PATH = r"E:\Transactions Fraud Detection\fraud_xgboost_model.joblib"
FEATURE_PATH = r"E:\Transactions Fraud Detection\feature_columns.json"

TOPIC = "transactions"
BOOTSTRAP_SERVERS = "localhost:9092"
FRAUD_THRESHOLD = 0.4

# Load the exact feature order used during training
with open(FEATURE_PATH, "r") as file:
    feature_columns = json.load(file)

# Load your existing trained XGBoost model
artifact = joblib.load(MODEL_PATH)

print("Loaded artifact type:", type(artifact))

model = artifact["model"]
feature_columns = artifact["feature_columns"]

print("Loaded classifier type:", type(model))
print("Has predict_proba:", hasattr(model, "predict_proba"))
print("Expected features:", len(feature_columns))
print("Fraud threshold:", FRAUD_THRESHOLD)

# Start the Kafka consumer
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="paysim-fraud-consumer",
    value_deserializer=lambda data: json.loads(data.decode("utf-8"))
)

print(f"Listening for transactions on topic: {TOPIC}")


try:
    for message in consumer:
        event = message.value

        # Skip older test messages that do not use the PaySim format
        if not isinstance(event, dict) or "features" not in event:
            print("Skipping test message: expected 'features' field not found.")
            continue

        features = event["features"]

        # Validate required model features
        missing = [col for col in feature_columns if col not in features]
        if missing:
            print(f"Skipping transaction: missing features {missing}")
            continue

        # Prepare features in the exact training order
        input_df = pd.DataFrame(
            [[features[col] for col in feature_columns]],
            columns=feature_columns
        )

        # Predict fraud probability
        fraud_probability = float(
            model.predict_proba(input_df)[0][1]
        )
        predicted_label = int(fraud_probability >= FRAUD_THRESHOLD)
        decision = "FRAUD" if predicted_label == 1 else "LEGITIMATE"

        print("\n------------------------------")
        print(f"Transaction ID: {event.get('transaction_id', 'Unknown')}")
        print(f"Transaction type: {event.get('transaction_type', 'Unknown')}")
        print(f"Amount: {features['amount']:.2f}")
        print(f"Fraud probability: {fraud_probability:.6f}")
        print(f"Predicted label: {predicted_label}")
        print(f"Decision: {decision}")

        if "actual_is_fraud" in event:
            print(f"Actual label: {event['actual_is_fraud']}")

except KeyboardInterrupt:
    print("\nConsumer stopped by user.")

finally:
    consumer.close()
    print("Kafka consumer closed.")
