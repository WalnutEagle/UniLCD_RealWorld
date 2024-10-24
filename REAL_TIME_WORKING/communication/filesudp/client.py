import socket
import pickle
import torch

# UDP host and port
HOST = '128.197.164.42'
PORT = 8083

def connect_to_udp_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return client_socket

def receive_file_udp(client_socket):
    while True:
        file_size, _ = client_socket.recvfrom(4096)  # Receive file size
        file_size = pickle.loads(file_size)
        if file_size == 0:
            break  # No more files to receive

        with open(f"received_file_{time.time()}.bin", 'wb') as f:
            received_size = 0
            while received_size < file_size:
                chunk, _ = client_socket.recvfrom(4096)
                f.write(chunk)
                received_size += len(chunk)
        print(f"File received successfully!")

def send_tensor_udp(client_socket, tensor):
    serialized_tensor = pickle.dumps(tensor)
    client_socket.sendto(pickle.dumps(len(serialized_tensor)), (HOST, PORT))  # Send length
    client_socket.sendto(serialized_tensor, (HOST, PORT))  # Send tensor
    print(f"Tensor {tensor} sent successfully!")

def udp_client_loop(client_socket):
    while True:
        # Optionally receive files
        receive_file_udp(client_socket)

        # Send tensor independently
        tensor_to_send = torch.rand(2, 2)  # Example tensor
        send_tensor_udp(client_socket, tensor_to_send)

# Usage
client_socket = connect_to_udp_server()
udp_client_loop(client_socket)
