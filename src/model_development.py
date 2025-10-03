from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_recall_curve, precision_score, classification_report, confusion_matrix
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

def train_and_save_model():
    # Save and Train the model
    # NN structure
    nn_Model = Sequential()
    nn_Model.add(InputLayer(input_shape=(X_train.shape[1],)))
    nn_Model.add(Dense(128, activation='relu'))
    nn_Model.add(Dense(128, activation='relu'))
    nn_Model.add(BatchNormalization())
    nn_Model.add(Dense(1, activation='sigmoid'))

    # Compilation
    nn_Model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    # Checkpoint definition
    checkpoint = ModelCheckpoint('nn_Model.keras', save_best_only=True)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Model fitting
    class_weights = {0: 1, 1: 15}
    history = nn_Model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping],
        class_weight=class_weights
    )
    
    return nn_Model

if __name__ == '__main__':
    import os
    import random as rn
    os.environ['PYTHONHASHSEED'] = '0'
    rn.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.experimental.enable_op_determinism()
    try:
        X_train = joblib.load('data/X_train.joblib')
        y_train = joblib.load('data/y_train.joblib')
        X_val = joblib.load('data/X_val.joblib')
        y_val = joblib.load('data/y_val.joblib')
    except FileNotFoundError:
        print("Error: Training data not found. Please run the preprocessing script first.")
    nn_Model = train_and_save_model()
    if nn_Model:
        nn_Model.save('models/neural_network_model.h5')
        print("\nNeural Network model trained and saved successfully.")
    y_probabilities_nn = nn_Model.predict(X_val)
precision_nn, recall_nn, _ = precision_recall_curve(y_val, y_probabilities_nn)
y_pred_val_nn = (y_probabilities_nn >= 0.5).astype(int)
print("\nNeural Network Classification Report on Validation Set (Default Threshold):")
print(classification_report(y_val, y_pred_val_nn, target_names=['Not Fraud', 'Fraud']))
print("\nNeural Network Confusion Matrix on Validation Set (Default Threshold):")
print(confusion_matrix(y_val, y_pred_val_nn))