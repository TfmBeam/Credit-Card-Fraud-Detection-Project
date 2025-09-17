# src/socket_client.py
import socket
import json
import requests

API_URL = "http://127.0.0.1:8000/predict"

def run_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 9999))
    print("Connected to simulator.")

    # Read socket as stream
    with client_socket.makefile("r") as stream:
        for line in stream:
            if not line.strip():
                continue

            transaction = json.loads(line.strip())

            # Send transaction to FastAPI
            response = requests.post(API_URL, json=transaction)
            print("Transaction:", transaction)
            print("Prediction:", response.json())

if __name__ == "__main__":
    run_client()
