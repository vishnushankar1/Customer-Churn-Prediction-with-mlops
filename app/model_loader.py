# app/model_loader.py

import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load scaler
scaler_path = os.path.join("artifacts", "transformation", "scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Load model
model_path = os.path.join("artifacts", "model_training", "model.h5")
model = load_model(model_path)

def make_prediction(data: np.ndarray) -> str:
    data_scaled = scaler.transform([data])
    pred = model.predict(data_scaled)[0][0]
    return "Churn: Yes" if pred > 0.5 else "Churn: No"
