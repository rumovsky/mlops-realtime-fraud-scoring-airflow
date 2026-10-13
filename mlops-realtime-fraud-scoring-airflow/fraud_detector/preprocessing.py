import json
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.distance import great_circle


ARTIFACTS = json.loads((Path(__file__).parent / "preprocessing_artifacts.json").read_text())
CATEGORY_MAPS = ARTIFACTS["category_maps"]
MEAN_ENCODINGS = ARTIFACTS["mean_encodings"]
IMPUTER_MEANS = ARTIFACTS["imputer_means"]
FEATURE_ORDER = ARTIFACTS["feature_order"]
CAT_FEATURES = ARTIFACTS["cat_features"]


def _key(value):
    return "<NA>" if pd.isna(value) else str(value)


def preprocess(data: dict) -> pd.DataFrame:
    frame = pd.DataFrame([data])
    frame = frame.drop(columns=["name_1", "name_2", "street", "post_code"])
    for column in ["amount", "population_city", "lat", "lon", "merchant_lat", "merchant_lon"]:
        frame[column] = pd.to_numeric(frame[column])

    for column, mapping in CATEGORY_MAPS.items():
        frame[column + "_cat"] = frame[column].map(
            lambda value: mapping.get(_key(value), "cat_NAN")
        )
        frame = frame.drop(columns=column)

    timestamp = pd.to_datetime(frame["transaction_time"])
    frame["hour"] = timestamp.dt.hour
    frame["year"] = timestamp.dt.year
    frame["month"] = timestamp.dt.month
    frame["day_of_month"] = timestamp.dt.day
    frame["day_of_week"] = timestamp.dt.dayofweek
    frame = frame.drop(columns="transaction_time")

    mean_columns = list(MEAN_ENCODINGS)
    for column in mean_columns:
        frame[column] = frame[column].fillna("cat_NAN")
        frame[column + "_mean_enc"] = frame[column].map(
            lambda value: MEAN_ENCODINGS[column].get(_key(value), np.nan)
        )

    frame["distance"] = frame.apply(
        lambda row: great_circle(
            (row["lat"], row["lon"]),
            (row["merchant_lat"], row["merchant_lon"]),
        ).km,
        axis=1,
    )
    frame = frame.drop(columns=["lat", "lon", "merchant_lat", "merchant_lon"])

    for column, mean in IMPUTER_MEANS.items():
        frame[column] = frame[column].fillna(mean)
        frame[column + "_log"] = np.log1p(frame[column])
        frame = frame.drop(columns=column)

    for column in CAT_FEATURES:
        frame[column] = frame[column].fillna("cat_NAN").astype(str)

    return frame[FEATURE_ORDER]
