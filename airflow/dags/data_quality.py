import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/tasks")
from pipeline import ensure_source
from dq_tasks import write_report, assert_quality


with DAG(
    dag_id="data_quality",
    start_date=datetime(2026, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
) as dag:
    source = PythonOperator(task_id="initialize_store", python_callable=ensure_source)
    report = PythonOperator(
        task_id="write_html_report", python_callable=write_report,
        op_kwargs={"run_id": "{{ run_id }}"},
    )
    verify = PythonOperator(
        task_id="assert_quality", python_callable=assert_quality,
        op_kwargs={"run_id": "{{ run_id }}"},
    )
    source >> report >> verify
