import pandas as pd
import numpy as np

def calculate_risk(row):
    risk = 0

    # -------------------------
    # Oxygen Risk
    # -------------------------
    if row["spo2_mean"] < 92:
        risk += 30
    elif row["spo2_mean"] < 95:
        risk += 15

    # -------------------------
    # Heart Rate Trend Risk
    # -------------------------
    if row["hr_slope"] > 0.5:
        risk += 20
    if row["hr_mean"] > 110:
        risk += 15

    # -------------------------
    # Blood Pressure Instability
    # -------------------------
    if row["bp_sys_std"] > 10:
        risk += 20

    # -------------------------
    # Overall Instability
    # -------------------------
    if row["instability_index"] > 15:
        risk += 15

    # -------------------------
    # ML Anomaly Score Contribution
    # -------------------------
    if row["anomaly_flag"]:
        risk += 25

    return min(risk, 100)


def calculate_confidence(row):
    # Confidence is lower if motion is high (sensor reliability ↓)
    confidence = 1.0

    if row["motion_mean"] > 0.8:
        confidence -= 0.3

    if row["anomaly_score"] < -0.3:
        confidence += 0.1

    return max(min(confidence, 1.0), 0.1)


def apply_risk_scoring(df):
    df["risk_score"] = df.apply(calculate_risk, axis=1)
    df["confidence"] = df.apply(calculate_confidence, axis=1)
    return df


if __name__ == "__main__":
    df = pd.read_csv("anomaly_results.csv")
    df = apply_risk_scoring(df)
    df.to_csv("final_risk_output.csv", index=False)
    print("Risk scoring complete → final_risk_output.csv")
