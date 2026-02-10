import pandas as pd
from sklearn.metrics import precision_score, recall_score

def create_ground_truth(df):
    df["true_emergency"] = (
        (df["spo2_mean"] < 92) |
        (df["hr_mean"] > 115)
    )
    return df


def evaluate_alerts(df):
    y_true = df["true_emergency"]
    y_pred = df["risk_score"] > 60   # alert threshold

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    false_alert_rate = ((y_pred == 1) & (y_true == 0)).sum() / len(df)

    return precision, recall, false_alert_rate


def calculate_alert_latency(df):
    emergency_indices = df.index[df["true_emergency"] == True]

    if len(emergency_indices) == 0:
        return None

    first_emergency = emergency_indices[0]

    alert_indices = df.index[df["risk_score"] > 60]

    early_alerts = alert_indices[alert_indices < first_emergency]

    if len(early_alerts) > 0:
        latency = first_emergency - early_alerts[0]
        return latency  # in windows (~30 sec each)

    return None


if __name__ == "__main__":
    df = pd.read_csv("final_risk_output.csv")

    df = create_ground_truth(df)
    precision, recall, far = evaluate_alerts(df)
    latency = calculate_alert_latency(df)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"False Alert Rate: {far:.2f}")
    print(f"Alert Latency (windows): {latency}")
