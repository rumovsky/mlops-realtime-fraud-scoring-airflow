"""Tasks used by the model retraining DAG."""

import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, "/opt/airflow/project/fraud_detector")
from preprocessing import CAT_FEATURES, preprocess


DATA_PATH = Path(os.getenv("TRAIN_DATA_PATH", "/opt/airflow/data/train.csv"))
WORK_DIR = Path("/opt/airflow/work")
MODEL_PATH = Path("/opt/airflow/models/model.cbm")
VERSION_PATH = Path("/opt/airflow/models/model.version")
STAGED_MODEL_PATH = MODEL_PATH.with_name(MODEL_PATH.name + ".tmp")
SAMPLE_SIZE = int(os.getenv("TRAINING_ROWS", "5000"))


def prepare_training_sample() -> None:
    """Read a bounded CSV sample and write it for the next task.

    The input is the mounted training CSV. The output is overwritten so a
    retry starts from the same source and does not append duplicate rows.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data was not found: {DATA_PATH}")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    output_path = WORK_DIR / "train_sample.csv"
    with DATA_PATH.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "target" not in reader.fieldnames:
            raise ValueError("Training CSV must contain a target column")
        with output_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for index, row in enumerate(reader):
                writer.writerow(row)
                if index + 1 >= SAMPLE_SIZE:
                    break


def train_model() -> None:
    """Train a small CPU CatBoost model and save an atomic staging file."""
    sample_path = WORK_DIR / "train_sample.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Prepared sample was not found: {sample_path}")

    frame = pd.read_csv(sample_path)
    targets = frame.pop("target").astype(int)
    features = pd.concat(
        [preprocess(row) for row in frame.to_dict(orient="records")],
        ignore_index=True,
    )
    for column in CAT_FEATURES:
        features[column] = features[column].astype(str)

    model = CatBoostClassifier(
        iterations=30,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        random_seed=42,
        thread_count=1,
        verbose=False,
    )
    model.fit(features, targets, cat_features=CAT_FEATURES)
    model.save_model(STAGED_MODEL_PATH)


def publish_model() -> None:
    """Atomically publish the trained model and its version marker."""
    staged_path = STAGED_MODEL_PATH
    if not staged_path.exists():
        raise FileNotFoundError(f"Staged model was not found: {staged_path}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_path, MODEL_PATH)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    VERSION_PATH.write_text(version + "\n", encoding="utf-8")
