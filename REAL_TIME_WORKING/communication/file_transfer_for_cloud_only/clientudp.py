import socket
import pickle
import torch

# UDP host and port
HOST = '128.197.164.42'
PORT = 8083

def connect_to_udp_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return client_socket

def request_file_udp(client_socket, server_addr, file_name):
    client_socket.sendto(file_name.encode(), server_addr)  # Send file request
    file_size, _ = client_socket.recvfrom(4096)  # Receive file size
    file_size = pickle.loads(file_size)
    if file_size:
        with open(f"received_{file_name}", 'wb') as f:
            received_size = 0
            while received_size < file_size:
                chunk, _ = client_socket.recvfrom(4096)
                f.write(chunk)
                received_size += len(chunk)
        print(f"File {file_name} received successfully!")
    else:
        print("File not found on server.")

def send_tensor_udp(client_socket, server_addr, tensor):
    serialized_tensor = pickle.dumps(tensor)
    client_socket.sendto(pickle.dumps(len(serialized_tensor)), server_addr)  # Send length
    client_socket.sendto(serialized_tensor, server_addr)  # Send tensor
    print(f"Tensor {tensor} sent successfully!")

def udp_client_loop(client_socket):
    server_addr = (HOST, PORT)
    while True:
        file_request = input("Enter file name to request or 'q' to quit: ")
        if file_request == 'q':
            client_socket.sendto(file_request.encode(), server_addr)
            break
        request_file_udp(client_socket, server_addr, file_request)
        tensor_to_send = torch.rand(2, 2)  # Example tensor
        send_tensor_udp(client_socket, server_addr, tensor_to_send)

# Usage
client_socket = connect_to_udp_server()
udp_client_loop(client_socket)
