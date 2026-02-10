# 🏥 Smart Ambulance AI – Risk Detection System

---

## 📌 Project Overview

This project builds a decision-support AI system for a smart ambulance platform.  
The system monitors patient vitals in real time, detects early signs of deterioration, and provides a risk score with confidence.

**Objective:**  
Develop a robust machine learning pipeline that works with noisy, safety-critical time-series data.

---

## ❤️ Vitals Used

- Heart Rate (HR)  
- SpO₂ (Oxygen Saturation)  
- Blood Pressure (Systolic/Diastolic)  
- Motion/Vibration Signal  

---

## ⚙️ System Pipeline

Synthetic Data Generation  
→ Artifact Detection & Cleaning  
→ Feature Engineering (30s windows)  
→ Anomaly Detection Model  
→ Risk Scoring Logic  
→ Evaluation Metrics  
→ FastAPI Service  
→ Dashboard UI (Bonus)

---

## 📊 Data Simulation

Synthetic time-series vitals simulate:

- Normal transport  
- Gradual deterioration  
- Motion artifacts (sensor noise)  
- Recovery phase  
- Missing data segments  

---

## 🧹 Artifact Handling

Noise sources addressed before ML:

- SpO₂ false drops removed using motion detection  
- HR spikes smoothed using filtering  
- Missing BP interpolated  
- Physiological bounds clipping applied  

---

## 🧠 Feature Engineering

Sliding window = 30 seconds

Extracted features:

- Mean and standard deviation  
- Signal slope (trend detection)  
- Variability measures  
- Motion statistics  
- Instability index  

This enables early detection of gradual deterioration.

---

## 🤖 Anomaly Detection

Model used: Isolation Forest

Reason:

- Effective for unlabeled abnormal patterns  
- Detects deviations from normal physiological behavior  

Outputs:

- Anomaly score  
- Anomaly flag  

---

## 🩺 Risk Scoring Logic

Clinical risk score combines:

- Low SpO₂  
- Rising HR  
- BP instability  
- Instability index  
- ML anomaly flag  

Also includes a confidence score reduced during high motion.

---

## 📈 Evaluation

Metrics reported:

- Precision  
- Recall  
- False alert rate  
- Alert latency  

**Design Principle:**  
In ambulance systems, recall is prioritized to avoid missing life-threatening deterioration.

---

## ⚠️ Failure Case Analysis

Three failure scenarios analyzed:

1. Slow oxygen drop missed  
2. False alert during high motion  
3. Missing BP data misinterpreted  

Each includes cause and mitigation strategies.

---

## 🌐 API Service

Built using FastAPI

Endpoint:

POST /predict

Input: Patient vitals  
Output:

- Anomaly flag  
- Risk score  
- Confidence score  

---

## 🖥️ Dashboard UI (Bonus)

Frontend dashboard provides:

- Vital input form  
- Risk visualization  
- Live HR & SpO₂ graph  
- Risk level colors  
- Alert sound for critical cases  

---

## 🏥 Safety-Critical Considerations

- Most dangerous failure: missed deterioration  
- False alert reduction via multi-signal confirmation  
- Final clinical decisions must never be fully automated  

System acts as decision support, not replacement for medical professionals.

---

## 📂 Repository Structure

src/ – ML pipeline code  
api/ – FastAPI service  
models/ – Saved model  
ui/ – Dashboard frontend  
data/ – Generated data  

---

## 🔁 Reproducibility Steps

1. Generate data  
2. Clean artifacts  
3. Create features  
4. Train model  
5. Run API  
6. Open dashboard  

---

## 🔥 Final Summary

This project demonstrates:

✔ Handling noisy time-series medical data  
✔ ML reasoning and anomaly detection  
✔ Clinical-style risk modeling  
✔ Safety-first AI thinking  
✔ Real-time ML deployment  

---

**Author:** Mohamed Aslam  
AI/ML Engineer Intern Candidate
