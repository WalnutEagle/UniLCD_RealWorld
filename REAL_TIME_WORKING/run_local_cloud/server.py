
# The size is too large sadly need to fix it down below 

import socket
import pickle
import torch  # For PyTorch tensor handling
import time

# Define host and port
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 8083       # Port to listen on

# Function to start the server and accept a connection
def start_server():
    start = time.time()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Server listening on {HOST}:{PORT}...")
    
    conn, addr = server_socket.accept()
    # print(f"Connected by {addr}")
    # print(f"Connection Created: {time.time()-start}seconds")
    return conn

# Function to receive data in chunks
def receive_data(conn):
    data_length = int.from_bytes(conn.recv(4), 'big')  # First, receive the length of the data
    data = b""
    while len(data) < data_length:
        packet = conn.recv(4096)  # Receive data in chunks
        if not packet:
            break
        data += packet
    received_data = pickle.loads(data)
    return received_data

# Function to send data with length prefix
def send_response(conn, response):
    data = pickle.dumps(response)
    data_length = len(data)
    conn.sendall(data_length.to_bytes(4, 'big'))  # Send the length of the data first
    conn.sendall(data)  # Then send the actual data

# Main server loop function for processing data
def server_loop(conn):
    while True:
        received_data = receive_data(conn)

        # If received_data is None, no data was received, so break the loop
        if received_data is None:
            print("No data received. Closing connection...")
            break
        
        # Handle text or tensor data
        if isinstance(received_data, str):
            print(f"Received text message: {received_data}")
            send_response(conn, "Text received!")
        elif isinstance(received_data, torch.Tensor):
            print(f"Received PyTorch tensor data: \n{received_data}")
            t1 = time.time()
            tensor_data = torch.rand(1, 4, 150, 130)
            send_response(conn, tensor_data)
            print(f"Tensor Sent, {time.time()-t1}seconds")
        else:
            print(f"Received unknown data type: {type(received_data)}")
            send_response(conn, "Unknown data type received!")
    
    conn.close()

# Usage example:
conn = start_server()
server_loop(conn)
