import socket
import os
import pickle
import time

# UDP host and port
HOST = '0.0.0.0'
PORT = 8083

def start_udp_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((HOST, PORT))
    print(f"UDP Server listening on {HOST}:{PORT}...")
    return server_socket

def send_file_udp(server_socket, client_addr, file_path):
    file_size = os.path.getsize(file_path)
    server_socket.sendto(pickle.dumps(file_size), client_addr)  # Send file size
    with open(file_path, 'rb') as f:
        while (chunk := f.read(4096)):  # Send file in chunks
            server_socket.sendto(chunk, client_addr)
    print(f"File {file_path} sent successfully!")

def receive_tensor_udp(server_socket):
    data, _ = server_socket.recvfrom(4096)
    data_length = pickle.loads(data)
    data = b""
    while len(data) < data_length:
        chunk, _ = server_socket.recvfrom(4096)
        data += chunk
    tensor = pickle.loads(data)
    return tensor

def udp_server_loop(server_socket):
    while True:
        request, client_addr = server_socket.recvfrom(4096)  # Receive request
        request = request.decode()
        if request == 'q':
            print(f"Client at {client_addr} closed connection.")
            break
        elif request.startswith('send_file:'):
            file_path = request.split(':')[1]
            if os.path.exists(file_path):
                send_file_udp(server_socket, client_addr, file_path)
            else:
                server_socket.sendto(b'File not found', client_addr)

def send_files(server_socket, client_addr):
    files_to_send = ['file1.txt', 'file2.txt', 'file3.txt']  # Update with your actual file paths
    for file_path in files_to_send:
        time.sleep(1)  # Optional delay between sends
        send_file_udp(server_socket, client_addr, file_path)

# Usage
server_socket = start_udp_server()
client_addr = ('<CLIENT_IP>', PORT)  # Replace with actual client IP
send_files(server_socket, client_addr)
udp_server_loop(server_socket)
