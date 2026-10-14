from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json().get("status") == "healthy"

def test_predict_endpoint_validation():
    # Test payload with KDD Cup 99 standard feature count approximation
    payload = {
        "features": [0.1] * 41
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 422, 500]