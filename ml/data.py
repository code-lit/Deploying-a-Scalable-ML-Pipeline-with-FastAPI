from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def clean_census_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Census dataset.

    - Strips whitespace from string columns (handles messy ', ' formatting)
    - Replaces '?' with NaN and drops rows with missing values
    - Normalizes the salary label by removing trailing '.' if present
    """
    df = df.copy()

    # Strip whitespace from object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # Replace missing marker and drop incomplete rows
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    # Normalize salary values if present
    if "salary" in df.columns:
        df["salary"] = df["salary"].astype(str).str.replace(".", "", regex=False).str.strip()

    return df


def process_data(
    X: pd.DataFrame,
    categorical_features: list[str],
    label: str | None = None,
    training: bool = True,
    encoder: OneHotEncoder | None = None,
    lb: LabelBinarizer | None = None,
):
    """Process data for model training/inference.

    When training=True, fits and returns `encoder` and `lb`.
    When training=False, uses provided `encoder` (and `lb` if label is present).

    Returns
    -------
    X_out : np.ndarray
    y_out : np.ndarray | np.array([])
    encoder : OneHotEncoder
    lb : LabelBinarizer
    """
    X = clean_census_df(X)

    # Separate label if provided
    if label is not None:
        y = X[label].values
        X = X.drop(columns=[label])
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features).values

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = encoder.fit_transform(X_categorical)

        lb = LabelBinarizer()
        y_out = lb.fit_transform(y).ravel() if label is not None else np.array([])
    else:
        if encoder is None:
            raise ValueError("encoder must be provided when training=False")

        X_cat = encoder.transform(X_categorical)

        if label is not None:
            if lb is None:
                raise ValueError("lb must be provided when label is not None and training=False")
            y_out = lb.transform(y).ravel()
        else:
            y_out = np.array([])

    X_out = np.concatenate([X_continuous, X_cat], axis=1)
    return X_out, y_out, encoder, lb


def apply_label(inference) -> str:
    """Convert binary prediction to salary label string."""
    val = int(np.array(inference).ravel()[0])
    return ">50K" if val == 1 else "<=50K"
