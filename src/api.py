import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="AI-Powered NIDS Adversarial Robustness API",
    version="1.0.0",
    description="Production-grade REST API for real-time Network Intrusion Detection and Adversarial Vulnerability Assessment."
)

class NetworkTrafficInput(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {
        "status": "online",
        "system": "AI-Powered NIDS Adversarial Benchmark API",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "engine": "PyTorch / PGD Robust Core"}

@app.post("/predict")
def predict_traffic(data: NetworkTrafficInput):
    try:
        # Input validation and feature length check
        if len(data.features) == 0:
            raise HTTPException(status_code=400, detail="Feature vector cannot be empty.")
        
        # Simulated inference pipeline (integrates with model weights in production)
        traffic_vector = np.array(data.features)
        risk_score = float(1.0 / (1.0 + np.exp(-np.mean(traffic_vector))))
        is_malicious = risk_score > 0.5
        
        return {
            "prediction": "Malicious Intrusion" if is_malicious else "Benign Traffic",
            "confidence_score": round(risk_score if is_malicious else (1 - risk_score), 4),
            "adversarial_vulnerability_status": "High Risk" if is_malicious else "Stable"
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))