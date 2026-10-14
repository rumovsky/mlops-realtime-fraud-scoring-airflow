import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/tasks")
from pipeline import ensure_source
from retrain_tasks import prepare_training_sample, train_model, compare_models, publish_model


with DAG(
    dag_id="retrain_fraud_model",
    start_date=datetime(2026, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(seconds=30)},
) as dag:
    source = PythonOperator(task_id="initialize_store", python_callable=ensure_source)
    prepare = PythonOperator(
        task_id="prepare_training_sample", python_callable=prepare_training_sample,
        op_kwargs={"run_id": "{{ run_id }}"},
    )
    train = PythonOperator(
        task_id="train_model", python_callable=train_model,
        op_kwargs={"run_id": "{{ run_id }}"},
    )
    compare = PythonOperator(
        task_id="compare_models", python_callable=compare_models,
        op_kwargs={"run_id": "{{ run_id }}"},
    )
    publish = PythonOperator(
        task_id="publish_model", python_callable=publish_model,
        op_kwargs={"run_id": "{{ run_id }}"},
    )
    source >> prepare >> train >> compare >> publish
