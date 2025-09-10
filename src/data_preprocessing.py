# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as rn
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data(df):
    df_copy = df.copy()
    scaler_amount = RobustScaler()
    df_copy['Amount'] = scaler_amount.fit_transform(df_copy['Amount'].to_numpy().reshape(-1,1))
    time = df_copy['Time']
    scaler_time = StandardScaler()
    df_copy['Time'] = scaler_time.fit_transform(df_copy['Time'].to_numpy().reshape(-1, 1))
    
    return df_copy, scaler_amount, scaler_time

def split_data(new_df, X,y, test_size=0.2,random_state = 35):
    X = new_df.drop('Class', axis = 1)
    y = new_df['Class']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.25, random_state = 35)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 35)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    import os
    # Assuming 'creditcard.csv' is in a 'data' folder
    df = pd.read_csv('data/creditcard.csv')
    df_processed, scaler_amount, scaler_time = preprocess_data(df)
    
    X = df_processed.drop('Class', axis=1)
    y = df_processed['Class']
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df,X, y)
    joblib.dump(X_train, 'data/X_train.joblib')
    joblib.dump(y_train, 'data/y_train.joblib')
    joblib.dump(X_val, 'data/X_val.joblib')
    joblib.dump(y_val, 'data/y_val.joblib')
    joblib.dump(X_test, 'data/X_test.joblib')
    joblib.dump(y_test, 'data/y_test.joblib')
    
    # Save the scalers as well, as the API will need them
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(scaler_amount, 'models/scaler_amount.joblib')
    joblib.dump(scaler_time, 'models/scaler_time.joblib')
    print("Scalers saved successfully.")
