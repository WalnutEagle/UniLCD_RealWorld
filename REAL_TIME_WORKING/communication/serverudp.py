import socket
import pickle
import torch  # For PyTorch tensor handling
import time

# Define host and port
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 8083      # Port to listen on

# Set up the server UDP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((HOST, PORT))

print(f"UDP Server listening on {HOST}:{PORT}...")

# Function to receive data from client
def receive_data():
    data, addr = server_socket.recvfrom(4096)
    received_data = pickle.loads(data)
    return received_data, addr

# Function to send a response back to client
def send_response(addr, response):
    server_socket.sendto(pickle.dumps(response), addr)

# Function to send large data in chunks
def send_large_data(addr, data):
    # Split data into chunks
    chunks = [data[i:i + 4096] for i in range(0, len(data), 4096)]
    for chunk in chunks:
        server_socket.sendto(pickle.dumps(chunk), addr)

import pickle
import struct

import pickle
import struct

def send_tensor(addr, tensor):
    tensor_bytes = pickle.dumps(tensor)  # Serialize tensor
    print(f"Sending tensor of size: {len(tensor_bytes)} bytes")

    # Pack the total size as a header
    total_size = len(tensor_bytes)
    size_header = struct.pack('!I', total_size)  # Big-endian unsigned int

    # Send the size header first
    try:
        server_socket.sendto(size_header, addr)
    except Exception as e:
        print(f"Failed to send size header: {e}")
        return

    # Send the tensor data in chunks
    chunk_size = 4096
    for i in range(0, total_size, chunk_size):
        try:
            server_socket.sendto(tensor_bytes[i:i + chunk_size], addr)
        except Exception as e:
            print(f"Failed to send data chunk: {e}")
            return

    print("Tensor sent successfully.")




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
            # Example of creating a new tensor to send back
            tensor_data = torch.rand(1, 4, 150, 130)
            print('now sending data')
            send_tensor(addr, tensor_data)  # Send tensor as bytes
            print('data sent')
        else:
            print(f"Received unknown data type: {type(received_data)}")
            send_response(addr, "Unknown data type received!")

# Usage:
server_loop()
