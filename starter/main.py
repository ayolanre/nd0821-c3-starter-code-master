"""FastAPI service for Census income predictions."""

import os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

if __package__:
    from .starter.ml.data import process_data
    from .starter.ml.model import inference
else:
    from starter.ml.data import process_data
    from starter.ml.model import inference

APP_DIR = Path(__file__).resolve().parent
model_path = Path(os.getenv("MODEL_PATH", "model/model.joblib")).expanduser()
MODEL_PATH = model_path if model_path.is_absolute() else APP_DIR / model_path
app = FastAPI(title="Census Income Predictor", version="1.0.0")
_artifact = None


class CensusInput(BaseModel):
    """One Census record. Aliases preserve the original CSV column names."""

    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(..., examples=[39])
    workclass: str = Field(..., examples=["State-gov"])
    fnlgt: int = Field(..., examples=[77516])
    education: str = Field(..., examples=["Bachelors"])
    education_num: int = Field(..., alias="education-num", examples=[13])
    marital_status: str = Field(
        ..., alias="marital-status", examples=["Never-married"]
    )
    occupation: str = Field(..., examples=["Adm-clerical"])
    relationship: str = Field(..., examples=["Not-in-family"])
    race: str = Field(..., examples=["White"])
    sex: str = Field(..., examples=["Male"])
    capital_gain: int = Field(..., alias="capital-gain", examples=[2174])
    capital_loss: int = Field(..., alias="capital-loss", examples=[0])
    hours_per_week: int = Field(..., alias="hours-per-week", examples=[40])
    native_country: str = Field(..., alias="native-country", examples=["United-States"])


def _get_artifact():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model artifact not found at {MODEL_PATH}. Run train_model.py first.",
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Census income prediction API"}


@app.post("/predict")
def predict(record: CensusInput) -> dict[str, str]:
    artifact = _get_artifact()
    row = pd.DataFrame([record.model_dump(by_alias=True)])
    features, _, _, _ = process_data(
        row,
        categorical_features=artifact["categorical_features"],
        training=False,
        encoder=artifact["encoder"],
        lb=artifact["label_binarizer"],
    )
    prediction = int(inference(artifact["model"], features)[0])
    label = artifact["label_binarizer"].classes_[prediction]
    return {"prediction": str(label)}
