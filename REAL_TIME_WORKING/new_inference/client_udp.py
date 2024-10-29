import socket
import pickle
import torch  # For PyTorch tensor handling
import time

# Define host and port (server address)
HOST = '128.197.164.42'  # Server's IP address (replace with actual IP for remote)
PORT = 8083              # Port to connect to

# Function to connect to the server
def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return client_socket

# Function to send data (text or tensor) to the server
def send_data(client_socket, data):
    serialized_data = pickle.dumps(data)
    data_length = len(serialized_data)
    # Send the length of the data first
    client_socket.sendto(data_length.to_bytes(4, 'big'), (HOST, PORT))
    # Then send the actual data
    client_socket.sendto(serialized_data, (HOST, PORT))

# Function to receive response from the server in chunks
def receive_response(client_socket):
    data_length, addr = client_socket.recvfrom(4)  # First, receive the length of the data
    data_length = int.from_bytes(data_length, 'big')
    data = b""
    while len(data) < data_length:
        packet, _ = client_socket.recvfrom(1400)  # Receive data in chunks
        data += packet
    return pickle.loads(data)


# Main client communication loop (can be called repeatedly)
def client_loop(client_socket):
    # while True:
    choice = 'n'
    
    if choice == 't':
        text_message = input("Enter your text message: ")
        send_data(client_socket, text_message)
        print("Server response:", receive_response(client_socket))
    
    elif choice == 'n':
        text_message = 'a'
        send_data(client_socket, text_message)
        print("Server response:", receive_response(client_socket))
        tensor_data = torch.rand(2, 2)
        # tensor_data = data  # Example PyTorch tensor data
        start = time.time()
        print(f"Sending PyTorch tensor: \n{tensor_data}")
        send_data(client_socket, tensor_data)
        print(f"Data sent in {(time.time()-start)*1000} Miliseconds")
        t1 = time.time()
        print("Server response:", receive_response(client_socket))
        print(f"Recived data in:{(time.time()-t1)*1000}Miliseconds.")
            
    elif choice == 'q':
        print("Closing connection...")
        # break

    client_socket.close()

# Usage example:
client_socket = connect_to_server()
client_loop(client_socket)
