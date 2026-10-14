import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/tasks")
from pipeline import ensure_source, append_month


with DAG(
    dag_id="append_labelled_month",
    start_date=datetime(2026, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(seconds=30)},
) as dag:
    initialize = PythonOperator(task_id="initialize_store", python_callable=ensure_source)
    append = PythonOperator(task_id="append_month", python_callable=append_month)
    initialize >> append
