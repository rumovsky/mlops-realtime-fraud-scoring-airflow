import json
import os
from pathlib import Path

from catboost import CatBoostClassifier

from features import preprocess
from preprocessing import preprocess as legacy_preprocess


MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "models"))
model = None
model_version = None
model_preprocessing = None
THRESHOLD = 0.98


def score_transaction(data):
    global model, model_version, model_preprocessing

    current = json.loads((MODEL_DIR / "current.json").read_text(encoding="utf-8"))
    if model is None or current["version"] != model_version:
        updated_model = CatBoostClassifier()
        updated_model.load_model(MODEL_DIR / current["file"])
        model = updated_model
        model_version = current["version"]
        model_preprocessing = current["preprocessing"]

    features = preprocess(data) if model_preprocessing == "v2" else legacy_preprocess(data)
    score = float(model.predict_proba(features, thread_count=1)[0, 1])
    return score, int(score > THRESHOLD), model_version
