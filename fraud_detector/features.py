"""Features shared by CPU training and scoring."""

import numpy as np
import pandas as pd


CAT_FEATURES = ["merch", "cat_id", "gender", "us_state", "jobs"]
NUMERIC_INPUTS = ["amount", "population_city", "lat", "lon", "merchant_lat", "merchant_lon"]
REQUIRED_FIELDS = ["transaction_time", *CAT_FEATURES, *NUMERIC_INPUTS]


def preprocess(data):
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame([data])
    result = frame[CAT_FEATURES].fillna("unknown").astype(str)
    numbers = frame[NUMERIC_INPUTS].apply(pd.to_numeric, errors="raise")
    timestamp = pd.to_datetime(frame["transaction_time"], format="mixed", errors="raise")
    result["amount_log"] = np.log1p(numbers["amount"])
    result["population_log"] = np.log1p(numbers["population_city"])
    result["hour"] = timestamp.dt.hour
    result["day_of_week"] = timestamp.dt.dayofweek
    result["month"] = timestamp.dt.month
    lat = np.radians(numbers["lat"])
    merchant_lat = np.radians(numbers["merchant_lat"])
    dlat = merchant_lat - lat
    dlon = np.radians(numbers["merchant_lon"] - numbers["lon"])
    haversine = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(merchant_lat) * np.sin(dlon / 2) ** 2
    result["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(haversine.clip(0, 1)))
    return result
