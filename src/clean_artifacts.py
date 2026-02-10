import pandas as pd
import numpy as np
from scipy.signal import medfilt

def clean_artifacts(df):
    df_clean = df.copy()

    # -------------------------------
    # 1️⃣ Handle Motion Artifacts (SpO₂ false drops)
    # If motion is high and SpO₂ drops suddenly → mark as artifact
    # -------------------------------
    motion_threshold = 0.8
    spo2_drop = df_clean["SpO2"].diff() < -5

    artifact_idx = (df_clean["motion"] > motion_threshold) & spo2_drop
    df_clean.loc[artifact_idx, "SpO2"] = np.nan

    # Interpolate cleaned SpO₂
    df_clean["SpO2"] = df_clean["SpO2"].interpolate()

    # -------------------------------
    # 2️⃣ Remove HR Spike Noise (road bumps)
    # Median filter smooths sudden unrealistic spikes
    # -------------------------------
    df_clean["HR"] = medfilt(df_clean["HR"], kernel_size=5)

    # -------------------------------
    # 3️⃣ Fill Missing Blood Pressure Values
    # -------------------------------
    df_clean["BP_sys"] = df_clean["BP_sys"].interpolate()
    df_clean["BP_dia"] = df_clean["BP_dia"].interpolate()

    # -------------------------------
    # 4️⃣ Clip Physiological Limits (safety bounds)
    # -------------------------------
    df_clean["HR"] = df_clean["HR"].clip(40, 180)
    df_clean["SpO2"] = df_clean["SpO2"].clip(70, 100)
    df_clean["BP_sys"] = df_clean["BP_sys"].clip(70, 200)
    df_clean["BP_dia"] = df_clean["BP_dia"].clip(40, 130)

    return df_clean


if __name__ == "__main__":
    df = pd.read_csv("ambulance_vitals.csv")
    df_clean = clean_artifacts(df)
    df_clean.to_csv("ambulance_vitals_cleaned.csv", index=False)
    print("Artifacts cleaned → ambulance_vitals_cleaned.csv")
