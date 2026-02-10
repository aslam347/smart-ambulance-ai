import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os


MODEL_PATH = "../models/anomaly_model.joblib"

def train_model(features_df):
    model = IsolationForest(
        n_estimators=150,
        contamination=0.05,   # expected anomaly %
        random_state=42
    )

    model.fit(features_df)
    return model


def predict_anomalies(model, features_df):
    scores = model.decision_function(features_df)  # anomaly score
    preds = model.predict(features_df)  # -1 = anomaly, 1 = normal

    result_df = features_df.copy()
    result_df["anomaly_score"] = scores
    result_df["anomaly_flag"] = preds == -1

    return result_df


if __name__ == "__main__":
    df = pd.read_csv("ambulance_features.csv")

    model = train_model(df)

    # ✅ Ensure models folder exists
    os.makedirs("../models/models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved")

    result_df = predict_anomalies(model, df)
    result_df.to_csv("anomaly_results.csv", index=False)

    print("Anomaly detection complete → anomaly_results.csv")

