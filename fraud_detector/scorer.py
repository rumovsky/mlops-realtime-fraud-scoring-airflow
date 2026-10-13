import os
from pathlib import Path

from catboost import CatBoostClassifier


MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "models" / "my_catboost.cbm"))
MODEL_VERSION_PATH = Path(
    os.getenv("MODEL_VERSION_PATH", Path(__file__).parent / "models" / "model.version")
)
model = None
model_mtime = None
model_version = "initial"
THRESHOLD = 0.98


def score_transaction(features):
    global model, model_mtime, model_version

    current_mtime = MODEL_PATH.stat().st_mtime_ns
    if model is None or current_mtime != model_mtime:
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        model_mtime = current_mtime
        if MODEL_VERSION_PATH.exists():
            model_version = MODEL_VERSION_PATH.read_text(encoding="utf-8").strip() or "unknown"

    score = float(model.predict_proba(features, thread_count=1)[0, 1])
    return score, int(score > THRESHOLD), model_version
