import socket
import pickle
import torch

HOST = '128.197.164.42'  # Server's IP address
PORT = 8083

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

# Receive data
received_data = b""
while True:
    part = client_socket.recv(4096)
    if not part:
        break
    received_data += part

# Deserialize the tensor
tensor = pickle.loads(received_data)
print(tensor)
print("Tensor received successfully.")
print("Received tensor shape:", tensor.shape)
client_socket.close()
