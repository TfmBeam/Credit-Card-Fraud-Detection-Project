import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from utils import load_model, load_scaler

# Directories
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
THRESH_PATH = os.path.join(MODELS_DIR, "best_threshold.joblib")


# --------------------
# Plot helpers
# --------------------
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns = __import__("seaborn")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc_curve(y_true, y_prob, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall(y_true, y_prob, model_name="Model"):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()


def full_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, digits=4))


# --------------------
# Data loader
# --------------------
def load_saved_data():
    """Load test split saved by data_preprocessing.py"""
    X_test = joblib.load(os.path.join(DATA_DIR, "X_test.joblib"))
    y_test = joblib.load(os.path.join(DATA_DIR, "y_test.joblib"))
    return X_test, y_test


# --------------------
# Robust threshold finder using F-beta
# --------------------
def find_best_threshold(y_true, y_prob, beta=2.0):
    """Find threshold maximizing F-beta but restrict to [0.05, 0.95]."""
    grid = np.linspace(0.05, 0.95, 91)  # steps of 0.01
    best_thresh, best_f = 0.5, 0
    for t in grid:
        y_pred = (y_prob > t).astype(int)
        if y_pred.sum() == 0:  # skip if no positives predicted
            continue
        # compute precision & recall
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-9)
        if f_beta > best_f:
            best_thresh, best_f = t, f_beta
    return best_thresh, best_f



# --------------------
# Main evaluation
# --------------------
def evaluate_model(save_threshold=True, show_plots=True, beta=2.0):
    # Load test data
    X_test, y_test = load_saved_data()

    # Load scalers
    scaler_time = load_scaler("scaler_time.joblib")
    scaler_amount = load_scaler("scaler_amount.joblib")

    # Scale features
    X_test = np.array(X_test, dtype=float)
    X_test[:, 0] = scaler_time.transform(X_test[:, 0].reshape(-1, 1)).ravel()
    X_test[:, 1] = scaler_amount.transform(X_test[:, 1].reshape(-1, 1)).ravel()

    # Load model
    model = load_model("neural_network_model.keras", framework="keras")
    y_prob = model.predict(X_test).ravel()

    # Find optimal threshold (F-beta)
    best_thresh, best_fbeta = find_best_threshold(y_test, y_prob, beta=beta)
    print(f"\nOptimal threshold (by F{beta}): {best_thresh:.4f} | Best F{beta}: {best_fbeta:.4f}")

    # Save threshold
    if save_threshold:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(best_thresh, THRESH_PATH)

    # Predictions
    y_pred = (y_prob > best_thresh).astype(int)

    # Metrics
    print("\n--- Neural Network Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    full_classification_report(y_test, y_pred)

    # Plots
    if show_plots:
        plot_confusion_matrix(y_test, y_pred, title="NN Confusion Matrix")
        plot_roc_curve(y_test, y_prob, model_name="Neural Network")
        plot_precision_recall(y_test, y_prob, model_name="Neural Network")

    return {
        "threshold": best_thresh,
        "best_fbeta": best_fbeta,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "y_test": y_test
    }


if __name__ == "__main__":
    evaluate_model()
