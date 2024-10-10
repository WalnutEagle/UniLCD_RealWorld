import socket
import pickle
import numpy as np

# Define host and port
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 65432      # Port to listen on

# Function to start the server and accept a connection
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Server listening on {HOST}:{PORT}...")
    
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    return conn

# Function to receive data from the client
def receive_data(conn):
    data = conn.recv(4096)
    if data:
        received_data = pickle.loads(data)
        return received_data
    return None

# Function to send response to the client
def send_response(conn, response):
    conn.sendall(pickle.dumps(response))

# Main server loop function for processing data
def server_loop(conn):
    while True:
        received_data = receive_data(conn)
        
        if not received_data:
            print("No data received. Closing connection...")
            break
        
        # Handle text or tensor data
        if isinstance(received_data, str):
            print(f"Received text message: {received_data}")
            send_response(conn, "Text received!")
        elif isinstance(received_data, np.ndarray):
            print(f"Received tensor data: \n{received_data}")
            send_response(conn, "Tensor received!")
    
    conn.close()

# Usage example:
# conn = start_server()
# server_loop(conn)
