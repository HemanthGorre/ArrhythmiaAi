# deploy/api_test.py
import requests
import numpy as np

X = np.load('data/processed/test_signals_6class.npy')[:3].tolist()  # First 3 samples
resp = requests.post("http://localhost:8000/predict", json={"windows": X})
print("Predicted labels:", resp.json())
