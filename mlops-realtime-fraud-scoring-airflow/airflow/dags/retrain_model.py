import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/tasks")
from retrain_tasks import prepare_training_sample, publish_model, train_model


with DAG(
    dag_id="retrain_fraud_model",
    start_date=datetime(2026, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["fraud", "retraining"],
) as dag:
    prepare = PythonOperator(
        task_id="prepare_training_sample",
        python_callable=prepare_training_sample,
    )
    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )
    publish = PythonOperator(
        task_id="publish_model",
        python_callable=publish_model,
    )

    prepare >> train >> publish
