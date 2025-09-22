import requests

BASE_URL = "http://127.0.0.1:8000"

def test_root():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.json()["message"] == "Fraud Detection API is running!"

def test_valid_prediction():
    payload = {"feature1": 0.5, "feature2": 1.2, "feature3": -0.8}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    assert "fraud_prediction" in response.json()

def test_missing_feature():
    payload = {"feature1": 0.5, "feature2": 1.2}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code in [400, 422, 500]  # depending on FastAPI validation
