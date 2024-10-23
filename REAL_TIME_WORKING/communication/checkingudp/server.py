'''import socket
import pickle
import torch  # For PyTorch tensor handling
import time

# Define host and port
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 8083       # Port to listen on

# Set up the server UDP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((HOST, PORT))

print(f"UDP Server listening on {HOST}:{PORT}...")

# Function to receive data from client
def receive_data():
    # Receive the length of the data first
    data_length, addr = server_socket.recvfrom(4)
    data_length = int.from_bytes(data_length, 'big')
    
    received_data = b""
    while len(received_data) < data_length:
        chunk, _ = server_socket.recvfrom(4096)
        received_data += chunk

    return pickle.loads(received_data), addr

# Function to send a response back to client
def send_response(addr, response):
    serialized_response = pickle.dumps(response)
    response_length = len(serialized_response)
    
    # Send the length of the response first
    server_socket.sendto(response_length.to_bytes(4, 'big'), addr)
    server_socket.sendto(serialized_response, addr)

# Main loop for server
def server_loop():
    while True:
        # Receive data and client address
        received_data, addr = receive_data()
        
        # Handle received data
        if isinstance(received_data, str):
            print(f"Received text message from {addr}: {received_data}")
            send_response(addr, "Text received!")
        elif isinstance(received_data, torch.Tensor):
            print(f"Received PyTorch tensor from {addr}: \n{received_data}")
            tensor_data = torch.rand(1, 4, 150, 130)  # Example tensor to send back
            send_response(addr, tensor_data)
        else:
            print(f"Received unknown data type from {addr}: {type(received_data)}")
            send_response(addr, "Unknown data type received!")

# Usage:
server_loop()
'''

import socket
import pickle
import torch  # For PyTorch tensor handling
import time

# Define host and port
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 8083       # Port to listen on

# Set up the server UDP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((HOST, PORT))

print(f"UDP Server listening on {HOST}:{PORT}...")

# Function to receive data from client
def receive_data():
    data_length, addr = server_socket.recvfrom(4)
    data_length = int.from_bytes(data_length, 'big')
    
    received_data = b""
    while len(received_data) < data_length:
        chunk, _ = server_socket.recvfrom(4096)
        received_data += chunk

    return pickle.loads(received_data), addr

# Function to send a response back to client
def send_response(addr, response):
    serialized_response = pickle.dumps(response)
    response_length = len(serialized_response)
    
    # Send the length of the response first
    server_socket.sendto(response_length.to_bytes(4, 'big'), addr)

    # Check if the response is too large
    if response_length > 4096:
        # Split the response into chunks
        for i in range(0, response_length, 4096):
            server_socket.sendto(serialized_response[i:i + 4096], addr)
    else:
        server_socket.sendto(serialized_response, addr)

# Main loop for server
def server_loop():
    while True:
        received_data, addr = receive_data()
        
        # Handle received data
        if isinstance(received_data, str):
            print(f"Received text message from {addr}: {received_data}")
            send_response(addr, "Text received!")
        elif isinstance(received_data, torch.Tensor):
            print(f"Received PyTorch tensor from {addr}: \n{received_data}")
            # Example of creating a new tensor to send back
            tensor_data = torch.rand(1, 4, 150, 130)  # Example tensor
            send_response(addr, tensor_data)
        else:
            print(f"Received unknown data type from {addr}: {type(received_data)}")
            send_response(addr, "Unknown data type received!")

# Usage:
server_loop()
