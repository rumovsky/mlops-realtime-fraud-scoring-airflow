#!/usr/bin/env bash
set -e

airflow db migrate
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com 2>/dev/null || true

airflow scheduler &
exec airflow webserver --port 8080
