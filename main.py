from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


PROJECT_PATH = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_PATH / "model"

encoder = load_model(MODEL_DIR / "encoder.pkl")
lb = load_model(MODEL_DIR / "lb.pkl")
model = load_model(MODEL_DIR / "model.pkl")

app = FastAPI()


@app.get("/")
async def get_root():
    return {
        "message": (
            "Welcome! Use POST /data/ for salary prediction (<=50K or >50K)."
        )
    }


@app.post("/data/")
async def post_inference(data: Data):
    data_dict = data.model_dump(by_alias=True)
    df = pd.DataFrame([data_dict])

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

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    pred = inference(model, X)
    return {"result": apply_label(pred)}