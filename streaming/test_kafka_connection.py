
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)

producer.send(
    "transactions",
    {
        "step": 1,
        "type": "TRANSFER",
        "amount": 5000.0,
        "isFraud": 0
    }
)

producer.flush()
producer.close()

print("Successfully sent test transaction to Kafka!")
