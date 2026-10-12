import json
import os
from kafka import KafkaConsumer, KafkaProducer

from preprocessing import preprocess
from scorer import score_transaction


BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "transactions")
OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "scoring")
consumer = KafkaConsumer(
    INPUT_TOPIC,
    bootstrap_servers=BROKER,
    group_id="fraud-detector",
    auto_offset_reset="earliest",
    enable_auto_commit=False,
    max_poll_records=1,
    value_deserializer=lambda value: json.loads(value.decode("utf-8")),
)
producer = KafkaProducer(
    bootstrap_servers=BROKER,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)

for message in consumer:
    payload = message.value
    transaction_id = payload["transaction_id"]
    features = preprocess(payload["data"])
    score, fraud_flag = score_transaction(features)
    producer.send(
        OUTPUT_TOPIC,
        {
            "transaction_id": transaction_id,
            "score": score,
            "fraud_flag": fraud_flag,
        },
    ).get(timeout=30)
    consumer.commit()
    print(f"{transaction_id}: score={score:.6f}, fraud_flag={fraud_flag}", flush=True)
