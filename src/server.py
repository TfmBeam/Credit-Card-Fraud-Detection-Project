# src/socket_simulator.py
import socket
import time
import joblib
import json
import pandas as pd

def run_simulator():
    # Load test dataset
    X_test = joblib.load("data/X_test.joblib")

    # Create server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("localhost", 9999))
    server_socket.listen(1)

    print("Simulator running on port 9999... Waiting for client connection.")
    conn, addr = server_socket.accept()
    print(f"Client connected: {addr}")

    try:
        for i, row in X_test.iterrows():
            transaction = row.to_dict()
            message = json.dumps(transaction) + "\n"
            conn.sendall(message.encode())

            print(f"Sent transaction {i}")
            time.sleep(1)  # delay between transactions
    finally:
        conn.close()
        server_socket.close()

if __name__ == "__main__":
    run_simulator()
