import pandas as pd
import numpy as np

WINDOW_SIZE = 30   # 30 seconds

def compute_slope(signal):
    x = np.arange(len(signal))
    slope = np.polyfit(x, signal, 1)[0]
    return slope

def create_features(df):
    feature_rows = []

    for i in range(0, len(df) - WINDOW_SIZE):
        window = df.iloc[i:i+WINDOW_SIZE]

        features = {
            # ---- Heart Rate Features ----
            "hr_mean": window["HR"].mean(),
            "hr_std": window["HR"].std(),
            "hr_slope": compute_slope(window["HR"]),

            # ---- Oxygen Features ----
            "spo2_mean": window["SpO2"].mean(),
            "spo2_std": window["SpO2"].std(),
            "spo2_slope": compute_slope(window["SpO2"]),

            # ---- Blood Pressure ----
            "bp_sys_mean": window["BP_sys"].mean(),
            "bp_sys_std": window["BP_sys"].std(),

            # ---- Motion ----
            "motion_mean": window["motion"].mean(),
            "motion_std": window["motion"].std(),

            # ---- Instability Index ----
            "instability_index": (
                window["HR"].std() +
                window["SpO2"].std() +
                window["BP_sys"].std()
            )
        }

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


if __name__ == "__main__":
    df = pd.read_csv("ambulance_vitals_cleaned.csv")
    features_df = create_features(df)
    features_df.to_csv("ambulance_features.csv", index=False)
    print("Feature engineering complete → ambulance_features.csv")
