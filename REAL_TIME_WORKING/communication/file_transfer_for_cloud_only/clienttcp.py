import socket
import pickle
import torch

# TCP host and port
HOST = '128.197.164.42'
PORT = 8083

def connect_to_tcp_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    return client_socket

# Function to request file from server
def request_file(client_socket, file_name):
    client_socket.sendall(file_name.encode())  # Send file request
    file_size = pickle.loads(client_socket.recv(4096))  # Receive file size
    if file_size:
        with open(f"received_{file_name}", 'wb') as f:
            received_size = 0
            while received_size < file_size:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                f.write(chunk)
                received_size += len(chunk)
        print(f"File {file_name} received successfully!")
    else:
        print("File not found on server.")

# Function to send tensor to server
def send_tensor(client_socket, tensor):
    serialized_tensor = pickle.dumps(tensor)
    client_socket.sendall(pickle.dumps(len(serialized_tensor)))  # Send length first
    client_socket.sendall(serialized_tensor)  # Send tensor
    print(f"Tensor {tensor} sent successfully!")

def tcp_client_loop(client_socket):
    while True:
        file_request = input("Enter file name to request or 'q' to quit: ")
        if file_request == 'q':
            client_socket.sendall(file_request.encode())
            break
        request_file(client_socket, file_request)
        tensor_to_send = torch.rand(2, 2)  # Example tensor
        send_tensor(client_socket, tensor_to_send)

    client_socket.close()

# Usage
client_socket = connect_to_tcp_server()
tcp_client_loop(client_socket)
