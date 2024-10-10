import socket
import pickle
import numpy as np

# Define host and port (server address)
HOST = '127.0.0.1'  # Server's IP address (replace with actual IP for remote)
PORT = 65432        # Port to connect to

# Function to connect to the server
def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    return client_socket

# Function to send data (text or tensor) to the server
def send_data(client_socket, data):
    client_socket.sendall(pickle.dumps(data))
    
# Function to receive response from the server
def receive_response(client_socket):
    response = client_socket.recv(4096)
    if response:
        return pickle.loads(response)
    return None

# Main client communication loop (can be called repeatedly)
def client_loop(client_socket):
    while True:
        choice = input("Enter 't' for text, 'n' for tensor, 'q' to quit: ")
        
        if choice == 't':
            text_message = input("Enter your text message: ")
            send_data(client_socket, text_message)
            print("Server response:", receive_response(client_socket))
            
        elif choice == 'n':
            tensor_data = np.random.rand(2, 2)  # Example tensor data
            print(f"Sending tensor: \n{tensor_data}")
            send_data(client_socket, tensor_data)
            print("Server response:", receive_response(client_socket))
            
        elif choice == 'q':
            print("Closing connection...")
            break

    client_socket.close()

# Usage example:
# client_socket = connect_to_server()
# client_loop(client_socket)
