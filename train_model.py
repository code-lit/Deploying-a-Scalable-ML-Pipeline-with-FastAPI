from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import clean_census_df, process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

PROJECT_PATH = Path(__file__).resolve().parent
DATA_PATH = PROJECT_PATH / "data" / "census.csv"
MODEL_DIR = PROJECT_PATH / "model"

# Load and clean the census.csv data (removes whitespace, '?' missing values)
data = pd.read_csv(DATA_PATH)
data = clean_census_df(data)

# Split the cleaned data
train, test = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["salary"],
)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
model = train_model(X_train, y_train)

# Save artifacts
save_model(model, MODEL_DIR / "model.pkl")
save_model(encoder, MODEL_DIR / "encoder.pkl")
save_model(lb, MODEL_DIR / "lb.pkl")

# Load model back (to demonstrate load_model)
model = load_model(MODEL_DIR / "model.pkl")

# Inference on test set
preds = inference(model, X_test)

# Metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Slice metrics
slice_path = PROJECT_PATH / "slice_output.txt"
slice_path.write_text("")

for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        sp, sr, sfb = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model,
        )
        with slice_path.open("a") as f:
            print(f"Precision: {sp:.4f} | Recall: {sr:.4f} | F1: {sfb:.4f}", file=f)
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)

print(f"Slice metrics saved to {slice_path.as_posix()}")
