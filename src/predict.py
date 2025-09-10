import numpy as np
import joblib
from utils import load_model, load_scaler
from sklearn.metrics import precision_recall_curve

def prepare_input(transaction, scaler_time, scaler_amount):
    """
    transaction: list or np.array with features in correct order.
                 Must include 'Time' at index 0 and 'Amount' at index 1.
    """
    X = np.array(transaction).reshape(1, -1)

    # scale Time and Amount
    X[:, 0] = scaler_time.transform(X[:, 0].reshape(-1, 1)).ravel()
    X[:, 1] = scaler_amount.transform(X[:, 1].reshape(-1, 1)).ravel()

    return X

def find_best_threshold(y_true, y_prob):
    """Find the probability threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = f1_scores.argmax()
    return thresholds[best_idx]

def predict_transaction(transaction, X_val=None, y_val=None):
    """
    Predict if a transaction is fraudulent.
    If validation data (X_val, y_val) is provided, 
    threshold is tuned using F1 score.
    Otherwise defaults to 0.5.
    """

    scaler_time = load_scaler("scaler_time.joblib")
    scaler_amount = load_scaler("scaler_amount.joblib")

    model = load_model("neural_network_model.keras", framework="keras")

    X = prepare_input(transaction, scaler_time, scaler_amount)
    prob = model.predict(X)[0][0]

    # Decide threshold
    if X_val is not None and y_val is not None:
        # Scale validation data
        X_val = np.array(X_val)
        X_val[:, 0] = scaler_time.transform(X_val[:, 0].reshape(-1, 1)).ravel()
        X_val[:, 1] = scaler_amount.transform(X_val[:, 1].reshape(-1, 1)).ravel()

        val_probs = model.predict(X_val).ravel()
        best_thresh = find_best_threshold(y_val, val_probs)
    else:
        best_thresh = 0.5

    return {
        "probability": float(prob),
        "prediction": int(prob > best_thresh),
        "threshold_used": float(best_thresh)
    }

if __name__ == "__main__":
    # Example usage
    # Load a sample transaction from your dataset
    sample = [10000, 250.0] + [0]*28  # replace with real values (Time, Amount, rest features)
    result = predict_transaction(sample)
    print(result)

