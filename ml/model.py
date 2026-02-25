from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def train_model(X_train, y_train):
    """Train and return a machine learning model."""
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """Compute precision, recall, and F1 (fbeta with beta=1)."""
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inference and return predictions."""
    return model.predict(X)


def save_model(obj, path):
    """Save a model/encoder/label binarizer to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)
    print(f"Model saved to {path.as_posix()}")


def load_model(path):
    """Load a model/encoder/label binarizer from disk."""
    path = Path(path)
    print(f"Loading model from {path.as_posix()}")
    with path.open("rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(
    data: pd.DataFrame,
    column_name: str,
    slice_value,
    categorical_features: list[str],
    label: str,
    encoder,
    lb,
    model,
):
    """Compute metrics on a data slice where `column_name == slice_value`."""
    data_slice = data[data[column_name] == slice_value].copy()
    if data_slice.shape[0] == 0:
        return 0.0, 0.0, 0.0

    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_slice)
    return compute_model_metrics(y_slice, preds)
