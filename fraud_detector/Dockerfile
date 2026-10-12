FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py preprocessing.py scorer.py .
COPY models/my_catboost.cbm models/my_catboost.cbm
COPY preprocessing_artifacts.json preprocessing_artifacts.json

CMD ["python", "app.py"]
