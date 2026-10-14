"""Prepare a time window, train, compare, and publish a versioned model."""

import hashlib
import sys
from datetime import date

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/opt/airflow/project/fraud_detector")
from features import CAT_FEATURES, preprocess
from preprocessing import preprocess as legacy_preprocess

from pipeline import (
    LOCK_ID, MODEL_DIR, WINDOW_MONTHS, connect, current_model,
    read_json, run_directory, shift_month, window_rows, write_json,
)


def prepare_training_sample(run_id):
    """Freeze train/test JSON for this run; retries reuse the saved snapshot.

    Reads the labelled store. Older WINDOW_MONTHS - 1 months form train;
    only the latest month forms test. No random split or target encoding is used.
    """
    work = run_directory(run_id)
    if (work / "window.json").exists():
        return
    current, latest, earliest, rows = window_rows()
    months = {date(timestamp.year, timestamp.month, 1) for _, timestamp, _, _ in rows}
    expected = {shift_month(earliest, n) for n in range(WINDOW_MONTHS)}
    if months != expected:
        raise ValueError(f"Expected consecutive months {expected}; got {months}")
    train, test = [], []
    for _, timestamp, data, target in rows:
        record = dict(data, target=target)
        (test if timestamp.date() >= latest else train).append(record)
    if {record["target"] for record in train} != {0, 1} or {record["target"] for record in test} != {0, 1}:
        raise ValueError("Both classes must be present in train and test")
    write_json(work / "train.json", train)
    write_json(work / "test.json", test)
    write_json(work / "window.json", {
        "previous_month": current.isoformat(),
        "first_month": earliest.isoformat(),
        "test_month": latest.isoformat(),
        "train_rows": len(train),
        "test_rows": len(test),
    })


def train_model(run_id):
    """Read frozen train rows, write a versioned CPU model without changing prod.

    The run ID determines the candidate filename; retries replace that candidate
    atomically and leave the active model pointer untouched.
    """
    work = run_directory(run_id)
    frame = pd.DataFrame(read_json(work / "train.json"))
    target = frame.pop("target").astype(int)
    version = "v2-" + hashlib.sha256(run_id.encode()).hexdigest()[:16]
    model = CatBoostClassifier(
        iterations=150,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        random_seed=42,
        thread_count=1,
        allow_writing_files=False,
        verbose=False,
        metadata={"model_version": version, "preprocessing": "v2"},
    )
    model.fit(preprocess(frame), target, cat_features=CAT_FEATURES)
    candidate = {"file": version + ".cbm", "version": version, "preprocessing": "v2"}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_DIR / (candidate["file"] + ".tmp")
    model.save_model(temporary)
    temporary.replace(MODEL_DIR / candidate["file"])
    write_json(work / "candidate.json", candidate)


def compare_models(run_id):
    """Evaluate candidate and active model on the same frozen latest month.

    Writes ROC-AUC and a publish decision. Retries recompute against the active
    version, preventing a comparison on a different test subset.
    """
    work = run_directory(run_id)
    frame = pd.DataFrame(read_json(work / "test.json"))
    target = frame.pop("target").astype(int)
    candidate = read_json(work / "candidate.json")
    incumbent = current_model()
    candidate_model = CatBoostClassifier()
    candidate_model.load_model(MODEL_DIR / candidate["file"])
    new_features = preprocess(frame)
    candidate_auc = float(roc_auc_score(target, candidate_model.predict_proba(new_features, thread_count=1)[:, 1]))
    active_model = CatBoostClassifier()
    active_model.load_model(MODEL_DIR / incumbent["file"])
    if incumbent["preprocessing"] == "v2":
        active_features = new_features
    else:
        active_features = pd.concat([legacy_preprocess(row) for row in frame.to_dict("records")], ignore_index=True)
    active_auc = float(roc_auc_score(target, active_model.predict_proba(active_features, thread_count=1)[:, 1]))
    decision = {
        "candidate_version": candidate["version"],
        "active_version": incumbent["version"],
        "candidate_roc_auc": candidate_auc,
        "active_roc_auc": active_auc,
        "accepted": candidate_auc >= active_auc,
        "test_month": read_json(work / "window.json")["test_month"],
    }
    write_json(work / "comparison.json", decision)
    print(decision)


def publish_model(run_id):
    """Activate an accepted candidate, then commit the trained-month pointer.

    Reads the comparison and candidate files. Publishes one atomic current.json
    pointer to an immutable model file. On rejection neither pointer moves.
    A retry after a crash between file publication and DB commit reconciles the
    month with the already published candidate, without adding another month.
    """
    work = run_directory(run_id)
    decision = read_json(work / "comparison.json")
    candidate = read_json(work / "candidate.json")
    window = read_json(work / "window.json")
    if not decision["accepted"]:
        print("Candidate rejected; active model and month pointer remain unchanged")
        return
    if not (MODEL_DIR / candidate["file"]).exists():
        raise FileNotFoundError(candidate["file"])
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
        active = current_model()
        if active["version"] not in {decision["active_version"], candidate["version"]}:
            raise RuntimeError("Active model changed since comparison; compare again")
        cursor.execute("SELECT current_month FROM training_state WHERE id = 1 FOR UPDATE")
        month = cursor.fetchone()[0]
        test_month = date.fromisoformat(window["test_month"])
        if month not in {date.fromisoformat(window["previous_month"]), test_month}:
            raise RuntimeError("Training month changed since the snapshot")
        write_json(MODEL_DIR / "current.json", dict(candidate, test_month=window["test_month"]))
        cursor.execute("""
            UPDATE training_state SET current_month = %s, model_version = %s WHERE id = 1
        """, (test_month, candidate["version"]))
    print(f"Published {candidate['version']} for {test_month}")
