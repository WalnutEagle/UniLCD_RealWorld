import socket
import pickle
import torch  # For PyTorch tensor handling

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

# Main loop for server
def server_loop():
    while True:
        # Receive data and client address
        received_data, addr = receive_data()
        
        if isinstance(received_data, str):
            print(f"Received text message from {addr}: {received_data}")
            send_response(addr, "Text received!")
        elif isinstance(received_data, torch.Tensor):
            print(f"Received PyTorch tensor from {addr}: \n{received_data}")
            send_response(addr, "Tensor received!")

# Usage:
# server_loop()
