import numpy as np
import pandas as pd

from ml.data import clean_census_df
from ml.model import compute_model_metrics, inference, train_model


def test_clean_census_df_strips_and_drops_missing():
    df = pd.DataFrame(
        {
            "workclass": [" Private ", " ? "],
            "education": [" HS-grad ", " Bachelors "],
            "salary": [" <=50K.", " >50K."],
        }
    )
    out = clean_census_df(df)
    assert len(out) == 1
    assert out.iloc[0]["workclass"] == "Private"
    assert out.iloc[0]["salary"] == "<=50K"


def test_train_and_inference_shapes():
    X = np.random.randn(50, 6)
    y = np.random.randint(0, 2, size=50)
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape[0] == X.shape[0]


def test_metrics_in_range():
    y = np.array([0, 1, 1, 0, 1])
    preds = np.array([0, 1, 0, 0, 1])
    p, r, f1 = compute_model_metrics(y, preds)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f1 <= 1.0
