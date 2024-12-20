
#### need to fix the size of the data send and recieved too mmuch below 

import socket
import pickle
import torch  # For PyTorch tensor handling
import time

# Define host and port (server address)
HOST = '128.197.164.42'  # Server's IP address (replace with actual IP for remote)
PORT = 8083              # Port to connect to

# Function to connect to the server
def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    return client_socket

# Function to send data (text or tensor) to the server
def send_data(client_socket, data):
    serialized_data = pickle.dumps(data)
    data_length = len(serialized_data)
    client_socket.sendall(data_length.to_bytes(4, 'big'))  # Send the length of the data first
    client_socket.sendall(serialized_data)  # Then send the actual data
    
# Function to receive response from the server in chunks
def receive_response(client_socket):
    data_length = int.from_bytes(client_socket.recv(4), 'big')  # First, receive the length of the data
    data = b""
    while len(data) < data_length:
        packet = client_socket.recv(4096)  # Receive data in chunks
        if not packet:
            break
        data += packet
    return pickle.loads(data)

# Main client communication loop (can be called repeatedly)
def client_loop(client_socket):
    while True:
        choice = input("Enter 't' for text, 'n' for tensor, 'q' to quit: ")
        
        if choice == 't':
            text_message = input("Enter your text message: ")
            send_data(client_socket, text_message)
            print("Server response:", receive_response(client_socket))
        
        elif choice == 'n':
            start = time.time()
            tensor_data = torch.rand(2, 2)  # Example PyTorch tensor data
            print(f"Sending PyTorch tensor: \n{tensor_data}")
            send_data(client_socket, tensor_data)
            print(f"Data sent in {time.time()-start} seconds")
            print("Server response:", receive_response(client_socket))
            
        elif choice == 'q':
            print("Closing connection...")
            break

    client_socket.close()

# Usage example:
# client_socket = connect_to_server()
# client_loop(client_socket)
