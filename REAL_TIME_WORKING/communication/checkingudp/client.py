import socket
import pickle
import torch
import time

# Define server address and port
HOST = '128.197.164.42'  # Replace with actual server IP for remote
PORT = 8083              # Server port

# Set up the client UDP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Function to send data (text or tensor) to server
def send_data(data):
    serialized_data = pickle.dumps(data)
    data_length = len(serialized_data)
    
    # Send the length of the data first
    client_socket.sendto(data_length.to_bytes(4, 'big'), (HOST, PORT))
    client_socket.sendto(serialized_data, (HOST, PORT))

# Function to receive response from server
def receive_response():
    # First, receive the length of the response
    response_length, _ = client_socket.recvfrom(4)
    response_length = int.from_bytes(response_length, 'big')
    
    received_data = b""
    while len(received_data) < response_length:
        chunk, _ = client_socket.recvfrom(4096)
        received_data += chunk
    
    return pickle.loads(received_data)

# Main client communication loop (similar to TCP version)
def client_loop():
    while True:
        choice = input("Enter 't' for text, 'n' for tensor, 'q' to quit: ")
        
        if choice == 't':
            text_message = input("Enter your text message: ")
            send_data(text_message)
            print("Server response:", receive_response())
        
        elif choice == 'n':
            start = time.time()
            tensor_data = torch.rand(2, 2)  # Example PyTorch tensor
            print(f"Sending PyTorch tensor: \n{tensor_data}")
            send_data(tensor_data)  # Send the tensor directly
            print(f"Data sent in {time.time()-start} seconds")
            print("Server response:", receive_response())
            
        elif choice == 'q':
            print("Closing connection...")
            break

    client_socket.close()

# Usage example:
client_loop()
