import json
import requests
from kafka import KafkaConsumer

from fraud_detection.features.engineering import TRANSACTION_FIELDS

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TOPIC = "transactions"
BOOTSTRAP_SERVERS = "localhost:9092"
API_URL = "http://127.0.0.1:8000/predict"

# --------------------------------------------------
# Start Kafka consumer
# --------------------------------------------------

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="paysim-fastapi-consumer",
    value_deserializer=lambda data: json.loads(
        data.decode("utf-8")
    )
)

print(f"Listening on Kafka topic: {TOPIC}")
print(f"Sending predictions to FastAPI: {API_URL}")

try:
    for message in consumer:
        event = message.value

        # Validate Kafka message
        if not isinstance(event, dict):
            print("Skipping invalid message.")
            continue

        features = event.get("features")

        if not isinstance(features, dict):
            print("Skipping message: missing features.")
            continue

        # FastAPI expects the six raw numeric fields plus "type".
        required_fields = TRANSACTION_FIELDS

        missing = [
            field for field in required_fields
            if field not in features
        ]

        if missing:
            print(f"Missing API input fields: {missing}")
            continue

        # Send transaction to the existing FastAPI service.
        transaction = {
            field: features[field]
            for field in required_fields
        }

        try:
            response = requests.post(
                API_URL,
                json=transaction,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()

        except requests.RequestException as exc:
            print(f"FastAPI request failed: {exc}")
            continue

        # Display the API's prediction.
        print("\n------------------------------")
        print(
            "Transaction ID:",
            event.get("transaction_id", "Unknown")
        )
        print(
            "Transaction type:",
            event.get("transaction_type", "Unknown")
        )
        print(f"Amount: {features['amount']:.2f}")
        print(
            "Fraud probability:",
            result["fraud_probability"]
        )
        print("Predicted label:", result["predicted_label"])
        print("Decision:", result["decision"])
        print("Threshold:", result["threshold"])

        if "actual_is_fraud" in event:
            print("Actual label:", event["actual_is_fraud"])

except KeyboardInterrupt:
    print("\nConsumer stopped by user.")

finally:
    consumer.close()
    print("Kafka consumer closed.")
