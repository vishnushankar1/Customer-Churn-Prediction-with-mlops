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
    data = np.array(data).reshape(1, -1)  # ✅ Fix: ensure input is 2D
    data_scaled = scaler.transform(data)  # ✅ Now scaler will not throw error
    pred = model.predict(data_scaled)[0][0]
    return "Churn: Yes" if pred > 0.5 else "Churn: No"
