import socket
import os
import pickle
import torch

# TCP host and port
HOST = '0.0.0.0'
PORT = 8083

# Function to start the server
def start_tcp_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"TCP Server listening on {HOST}:{PORT}...")
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    return conn

# Function to send file to the client
def send_file(conn, file_path):
    file_size = os.path.getsize(file_path)
    conn.sendall(pickle.dumps(file_size))  # Send file size first
    with open(file_path, 'rb') as f:
        while (chunk := f.read(4096)):  # Send file in chunks
            conn.sendall(chunk)
    print(f"File {file_path} sent successfully!")

# Function to receive tensor from the client
def receive_tensor(conn):
    data_length = pickle.loads(conn.recv(4096))
    data = b""
    while len(data) < data_length:
        data += conn.recv(4096)
    tensor = pickle.loads(data)
    return tensor

def tcp_server_loop(conn):
    while True:
        request = conn.recv(4096).decode()  # Client requests file or quits
        if request == 'q':
            print("Client closed connection.")
            break
        elif os.path.exists(request):  # Check if the file exists
            send_file(conn, request)
            tensor_response = receive_tensor(conn)  # Receive tensor from client
            print(f"Received tensor from client:\n{tensor_response}")
        else:
            conn.sendall(b'File not found')

    conn.close()

# Usage
conn = start_tcp_server()
tcp_server_loop(conn)
