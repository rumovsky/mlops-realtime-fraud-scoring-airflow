from pathlib import Path

from catboost import CatBoostClassifier


model = CatBoostClassifier()
model.load_model(Path(__file__).parent / "models" / "my_catboost.cbm")
THRESHOLD = 0.98


def score_transaction(features):
    score = float(model.predict_proba(features, thread_count=1)[0, 1])
    return score, int(score > THRESHOLD)
