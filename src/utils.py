import joblib
import tensorflow as tf
import pickle
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def load_scaler(name: str):
    return joblib.load(os.path.join(MODEL_DIR, name))

def load_model(model_name: str, framework="keras"):
    path = os.path.join(MODEL_DIR, model_name)
    if framework == "keras":
        return tf.keras.models.load_model(path)
    elif framework == "pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("Unsupported framework type")

def save_pickle(obj, filename: str):
    with open(os.path.join(MODEL_DIR, filename), "wb") as f:
        pickle.dump(obj, f)
