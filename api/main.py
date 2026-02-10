import sys
import os

# Add src folder to path FIRST
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

from risk_score import calculate_risk, calculate_confidence


# Create FastAPI app
app = FastAPI(title="Smart Ambulance AI Service")

# Enable CORS (for UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained ML model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'anomaly_model.joblib'))
model = joblib.load(MODEL_PATH)


class VitalInput(BaseModel):
    hr: float
    spo2: float
    bp_sys: float
    bp_dia: float
    motion: float


def create_feature_row(data: VitalInput):
    features = {
        "hr_mean": data.hr,
        "hr_std": 2,
        "hr_slope": 0.2,

        "spo2_mean": data.spo2,
        "spo2_std": 1,
        "spo2_slope": -0.1,

        "bp_sys_mean": data.bp_sys,
        "bp_sys_std": 5,

        "motion_mean": data.motion,
        "motion_std": 0.1,

        "instability_index": 5
    }
    return pd.DataFrame([features])


@app.post("/predict")
def predict(vitals: VitalInput):
    feature_df = create_feature_row(vitals)

    anomaly_score = model.decision_function(feature_df)[0]
    anomaly_flag = model.predict(feature_df)[0] == -1

    feature_df["anomaly_flag"] = anomaly_flag
    feature_df["anomaly_score"] = anomaly_score

    risk_score = calculate_risk(feature_df.iloc[0])
    confidence = calculate_confidence(feature_df.iloc[0])

    return {
        "anomaly": bool(anomaly_flag),
        "risk_score": round(risk_score, 2),
        "confidence": round(confidence, 2)
    }


@app.get("/")
def home():
    return {"message": "Smart Ambulance AI Service Running"}
