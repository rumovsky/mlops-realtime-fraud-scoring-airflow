# Realtime fraud scoring with Airflow

Проект развивает сервис скоринга из ДЗ1: Airflow по расписанию обучает новую модель и публикует её в общий каталог, а сервис скоринга подхватывает файл без перезапуска.

## Запуск

Нужны Docker и Docker Compose. Скачайте `train.csv` из соревнования Kaggle, указанного в задании, и положите его в `data/train.csv`.

```bash
docker compose up --build
```

Интерфейсы:

- Streamlit: `http://localhost:8501`
- Kafka UI: `http://localhost:8080`
- Airflow: `http://localhost:8081` (`admin` / `admin`)

В Airflow откройте DAG `retrain_fraud_model` и запустите его вручную. Он состоит из трёх задач:

1. `prepare_training_sample` — читает первые 5000 строк `data/train.csv`;
2. `train_model` — обучает небольшую CPU-модель CatBoost;
3. `publish_model` — атомарно заменяет `models/model.cbm` и обновляет `models/model.version`.

Также DAG запускается автоматически раз в 5 минут. После публикации новая транзакция, отправленная через Streamlit, получает в сообщении Kafka `scoring` поле `model_version`. Сервис перечитывает модель при изменении файла, поэтому перезапуск контейнера скоринга не нужен.

Каталог `data` предназначен для локального датасета и не входит в репозиторий.
