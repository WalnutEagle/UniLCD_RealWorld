import socket
import pickle
import torch  # For PyTorch tensor handling
import time

# Define server address and port
HOST = '128.197.164.42'  # Replace with actual server IP for remote
PORT = 8083              # Server port

# Set up the client UDP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Function to send data (text or tensor) to server
def send_data(data):
    client_socket.sendto(pickle.dumps(data), (HOST, PORT))

# Function to receive response from server
def receive_response():
    response, _ = client_socket.recvfrom(4096)
    return pickle.loads(response)

# Function to receive large data
def receive_large_data():
    received_data = b""
    while True:
        chunk, _ = client_socket.recvfrom(4096)
        if chunk == b"":  # Check for end of transmission
            break
        received_data += chunk
    # Check if we received any data before unpickling
    if received_data:
        return pickle.loads(received_data)
    else:
        raise ValueError("No data received before end of transmission")



# Example loop to send data
def client_loop():
    while True:
        choice = input("Enter 't' for text, 'p' for PyTorch tensor, 'q' to quit: ")

        if choice == 't':
            text_message = input("Enter your text message: ")
            send_data(text_message)
            print("Server response:", receive_response())
            
        elif choice == 'p':
            start = time.time()
            tensor_data = torch.rand(2, 2)  # Example PyTorch tensor
            print(f"Sending PyTorch tensor: \n{tensor_data}")
            send_data(tensor_data)  # Send as bytes
            print(f"Data sent in {time.time()-start} seconds")
            data = receive_large_data()
            print("Received tensor from server:\n", data)
            
        elif choice == 'q':
            print("Closing connection...")
            break

    client_socket.close()

# Usage:
client_loop()
