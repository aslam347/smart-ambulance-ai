import numpy as np
import pandas as pd

np.random.seed(42)

def simulate_patient(duration_minutes=30):
    seconds = duration_minutes * 60
    time = np.arange(seconds)

    # ---- Base Normal Signals ----
    hr = np.random.normal(80, 3, seconds)           # Heart Rate
    spo2 = np.random.normal(98, 1, seconds)         # Oxygen
    bp_sys = np.random.normal(120, 5, seconds)      # Systolic
    bp_dia = np.random.normal(80, 3, seconds)       # Diastolic
    motion = np.random.normal(0.2, 0.05, seconds)   # Low vehicle motion

    # ---- Phase 1: Normal (0–10 min) ----
    # already normal

    # ---- Phase 2: Gradual Deterioration (10–20 min) ----
    start = 10 * 60
    end = 20 * 60

    hr[start:end] += np.linspace(0, 40, end-start)       # HR rising
    spo2[start:end] -= np.linspace(0, 8, end-start)      # SpO2 dropping
    bp_sys[start:end] += np.linspace(0, 15, end-start)   # BP unstable

    # ---- Phase 3: Motion Artifacts (20–25 min) ----
    start = 20 * 60
    end = 25 * 60

    motion[start:end] = np.random.normal(1.2, 0.3, end-start)

    # Fake sensor issues during bumps
    spo2[start:end] -= np.random.choice([0, 15], size=end-start, p=[0.7, 0.3])
    hr[start:end] += np.random.choice([0, 50], size=end-start, p=[0.8, 0.2])

    # ---- Phase 4: Recovery (25–30 min) ----
    start = 25 * 60
    end = 30 * 60

    hr[start:end] -= np.linspace(30, 0, end-start)
    spo2[start:end] += np.linspace(5, 0, end-start)
    bp_sys[start:end] -= np.linspace(10, 0, end-start)

    # ---- Missing Data Simulation ----
    missing_idx = np.random.choice(seconds, 50, replace=False)
    bp_sys[missing_idx] = np.nan

    # ---- Build DataFrame ----
    df = pd.DataFrame({
        "timestamp_sec": time,
        "HR": hr,
        "SpO2": spo2,
        "BP_sys": bp_sys,
        "BP_dia": bp_dia,
        "motion": motion
    })

    return df


if __name__ == "__main__":
    df = simulate_patient()
    df.to_csv("ambulance_vitals.csv", index=False)
    print("Synthetic ambulance data generated: ambulance_vitals.csv")
