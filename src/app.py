from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from contextlib import asynccontextmanager

# -----------------------------
# Global placeholders
# -----------------------------
model = None
scaler_amount = None
scaler_time = None
best_threshold = 0.5

# -----------------------------
# Lifespan handler
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler_amount, scaler_time, best_threshold
    try:
        model = load_model("models/neural_network_model.h5",compile=False)
        scaler_amount = joblib.load("models/scaler_amount.joblib")
        scaler_time = joblib.load("models/scaler_time.joblib")
        best_threshold = joblib.load("models/best_threshold.joblib")
        print("Model and scalers loaded successfully")
    except Exception as e:
        print("Failed to load model/scalers:", e)

    yield  # Application runs here

    # (Optional) cleanup code goes here
    print("Shutting down API")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud detection using a trained Neural Network",
    version="1.0",
    lifespan=lifespan,
)

# -----------------------------
# Input Schema
# -----------------------------
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health_check():
    return {"msg": "API is running", "model_loaded": model is not None}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        return {"error": "Model not loaded"}

    # Convert input to numpy array
    features = np.array([
        transaction.Time,
        transaction.V1, transaction.V2, transaction.V3, transaction.V4,
        transaction.V5, transaction.V6, transaction.V7, transaction.V8,
        transaction.V9, transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15, transaction.V16,
        transaction.V17, transaction.V18, transaction.V19, transaction.V20,
        transaction.V21, transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27, transaction.V28,
        transaction.Amount
    ]).reshape(1, -1)  # shape (1, 30)

    # Scale Time and Amount
    try:
        features[0, 0] = scaler_time.transform([[features[0, 0]]])[0, 0]   
        features[0, -1] = scaler_amount.transform([[features[0, -1]]])[0, 0]  
    except Exception as e:
        print("Scaling failed:", e)

    # Predict probability
    prob = model.predict(features)[0][0]

    # Apply best threshold
    prediction = 1 if prob >= best_threshold else 0

    return {
        "probability": float(prob),
        "threshold": float(best_threshold),
        "prediction": int(prediction)
    }
