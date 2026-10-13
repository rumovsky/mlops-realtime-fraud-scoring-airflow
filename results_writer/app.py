import json
import os

import psycopg
from kafka import KafkaConsumer


BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://fraud:fraud@postgres:5432/fraud",
)


with psycopg.connect(POSTGRES_DSN, autocommit=True) as connection:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS scores (
            transaction_id TEXT PRIMARY KEY,
            score DOUBLE PRECISION NOT NULL,
            fraud_flag INTEGER NOT NULL,
            model_version TEXT NOT NULL DEFAULT 'unknown',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    connection.execute(
        "ALTER TABLE scores ADD COLUMN IF NOT EXISTS model_version TEXT NOT NULL DEFAULT 'unknown'"
    )

consumer = KafkaConsumer(
    "scoring",
    bootstrap_servers=BROKER,
    group_id="results-writer",
    auto_offset_reset="earliest",
    enable_auto_commit=False,
    max_poll_records=1,
    value_deserializer=lambda value: json.loads(value.decode("utf-8")),
)

with psycopg.connect(POSTGRES_DSN, autocommit=True) as connection:
    for message in consumer:
        result = message.value
        connection.execute(
            """
            INSERT INTO scores (transaction_id, score, fraud_flag, model_version)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (transaction_id) DO UPDATE
            SET score = EXCLUDED.score,
                fraud_flag = EXCLUDED.fraud_flag,
                model_version = EXCLUDED.model_version
            """,
            (
                result["transaction_id"],
                result["score"],
                result["fraud_flag"],
                result.get("model_version", "unknown"),
            ),
        )
        consumer.commit()
