from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import random

app = FastAPI(title="Fraud Detection A/B API")

MODEL_A_PATH = "models/fraud_model_vA.pkl"   
MODEL_B_PATH = "models/fraud_model_vB.pkl"   

model_A = joblib.load(MODEL_A_PATH)
model_B = joblib.load(MODEL_B_PATH)


class TransactionFeatures(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Time: float


@app.get("/")
def root():
    return {"message": "Fraud Detection A/B API is running!"}


@app.post("/predict")
def predict(features: TransactionFeatures):
    data = features.dict()
    df = pd.DataFrame([data])

    variant = random.choice(["A", "B"])

    if variant == "A":
        pred = model_A.predict(df)[0]
        prob = model_A.predict_proba(df)[0][1] if hasattr(model_A, "predict_proba") else None
    else:
        pred = model_B.predict(df)[0]
        prob = model_B.predict_proba(df)[0][1] if hasattr(model_B, "predict_proba") else None

    return {
        "variant": variant,
        "fraud_prediction": int(pred),
        "fraud_probability": float(prob) if prob is not None else None
    }
