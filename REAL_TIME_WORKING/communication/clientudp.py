import socket
import pickle
import torch  # For PyTorch tensor handling

# Define server address and port
HOST = '128.197.164.42'  # Replace with actual server IP for remote
PORT = 8083        # Server port

# Set up the client UDP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Function to send data (text or tensor) to server
def send_data(data):
    client_socket.sendto(pickle.dumps(data), (HOST, PORT))

# Function to receive response from server
def receive_response():
    response, _ = client_socket.recvfrom(4096)
    return pickle.loads(response)

# Example loop to send data
def client_loop():
    while True:
        choice = input("Enter 't' for text, 'p' for PyTorch tensor, 'q' to quit: ")

        if choice == 't':
            text_message = input("Enter your text message: ")
            send_data(text_message)
            print("Server response:", receive_response())
            
        elif choice == 'p':
            tensor_data = torch.rand(2, 2)  # Example PyTorch tensor
            print(f"Sending PyTorch tensor: \n{tensor_data}")
            send_data(tensor_data)
            print("Server response:", receive_response())
            
        elif choice == 'q':
            print("Closing connection...")
            break

    client_socket.close()

# Usage:
# client_loop()
