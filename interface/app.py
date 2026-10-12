import csv
import io
import json
import os
import uuid

import streamlit as st
from kafka import KafkaProducer
import psycopg


KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://fraud:fraud@postgres:5432/fraud",
)


st.set_page_config(page_title="Realtime fraud scoring", page_icon="✅")
st.title("Realtime fraud scoring")
st.write("Загрузите CSV и отправьте его строки отдельными сообщениями в Kafka.")

if st.button("Посмотреть результаты"):
    try:
        with psycopg.connect(POSTGRES_DSN) as connection:
            fraud_rows = connection.execute(
                """
                SELECT transaction_id, score, fraud_flag
                FROM scores
                WHERE fraud_flag = 1
                ORDER BY created_at DESC
                LIMIT 10
                """
            ).fetchall()
            recent_scores = connection.execute(
                """
                SELECT score
                FROM scores
                ORDER BY created_at DESC
                LIMIT 100
                """
            ).fetchall()
        st.subheader("Последние fraud-транзакции")
        st.dataframe(
            [
                {"transaction_id": row[0], "score": row[1], "fraud_flag": row[2]}
                for row in fraud_rows
            ],
            use_container_width=True,
        )
        st.subheader("Распределение скоров")
        counts = [0] * 10
        for (score,) in recent_scores:
            counts[min(int(score * 10), 9)] += 1
        st.bar_chart(
            {"score": [i / 10 + 0.05 for i in range(10)], "count": counts},
            x="score",
            y="count",
        )
    except Exception as error:
        st.error(f"Не удалось получить результаты: {error}")

uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        text_stream = io.TextIOWrapper(uploaded_file, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text_stream)
        fieldnames = reader.fieldnames
        preview = []
        for row in reader:
            preview.append(row)
            if len(preview) == 10:
                break
        text_stream.detach()
    except Exception as error:
        st.error(f"Не удалось прочитать CSV: {error}")
    else:
        st.success("OK: CSV-файл успешно загружен.")
        if not fieldnames:
            st.error("CSV-файл не содержит заголовка.")
            st.stop()
        st.write(f"Столбцов: {len(fieldnames)}")
        st.dataframe(preview, use_container_width=True)

        if st.button("Отправить строки в Kafka", type="primary"):
            try:
                producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BROKER,
                    value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
                )
                uploaded_file.seek(0)
                text_stream = io.TextIOWrapper(uploaded_file, encoding="utf-8-sig", newline="")
                reader = csv.DictReader(text_stream)
                sent_count = 0
                status = st.empty()
                for row in reader:
                    row_data = {
                        column: (None if value == "" else value)
                        for column, value in row.items()
                    }
                    producer.send(
                        KAFKA_TOPIC,
                        value={
                            "transaction_id": str(uuid.uuid4()),
                            "data": row_data,
                        },
                    )
                    sent_count += 1
                    if sent_count % 1000 == 0:
                        status.write(f"Подготовлено сообщений: {sent_count}")
                text_stream.detach()
                producer.flush()
                producer.close()
                status.write(f"Подготовлено сообщений: {sent_count}")
                st.success(f"Отправлено сообщений: {sent_count}")
            except Exception as error:
                st.error(f"Не удалось отправить сообщения в Kafka: {error}")
