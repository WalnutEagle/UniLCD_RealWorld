import socket
import pickle
import torch  # For PyTorch tensor handling
import time
import select
import numpy as np
# Define host and port
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 8083       # Port to listen on

# Function to start the server
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((HOST, PORT))
    print(f"Server listening on {HOST}:{PORT}...")
    return server_socket

# Function to receive data in chunks
def receive_data(server_socket):
    data_length, addr = server_socket.recvfrom(4)  # First, receive the length of the data
    data_length = int.from_bytes(data_length, 'big')
    data = b""
    while len(data) < data_length:
        packet, _ = server_socket.recvfrom(4096)  # Receive data in chunks
        data += packet
    received_data = pickle.loads(data)
    return received_data, addr

# Function to send data with length prefix
def send_response(server_socket, response, addr):
    data = pickle.dumps(response)
    data_length = len(data)
    
    # Send the length of the data first
    server_socket.sendto(data_length.to_bytes(4, 'big'), addr)

    # Split data into smaller chunks if it's too large
    chunk_size = 1400  # Set an appropriate chunk size
    for i in range(0, data_length, chunk_size):
        server_socket.sendto(data[i:i + chunk_size], addr)  # Send each chunk

# Main server loop function for processing data
def server_loop(server_socket):
    # while True:
    # time.sleep(1)
    # server_socket.listen(1)
    # receive_data, addr = server_socket.accept(1)
    # server_socket.setblocking(0)
    # ready = select.select([server_socket], [], [], 1)
    # if ready[0]:
    #     # data = mysocket.recv(4096)
    #     received_data, addr = receive_data(server_socket) 
    received_data, addr = receive_data(server_socket)  
    # Handle text or tensor data
    if isinstance(received_data, str):
        print(f"Received text message: {received_data} from {addr}")
        send_response(server_socket, "Text received!", addr)
    elif isinstance(received_data, torch.Tensor):
        s=time.time()
        print(f"Received PyTorch tensor data: \n{received_data} from {addr}")
        print(f"It took{(time.time()-s)*1000} Miliseconds.")
        tensor_data = np.zeros((1, 4, 150, 130)).tolist()
        # tensor_data = torch.rand(500, 500)
        print(tensor_data)
        t1 = time.time()
        send_response(server_socket, tensor_data, addr)
        print(f"Tensor Sent, {(time.time()-t1)*1000} Miliseconds")
    else:
        print(f"Received unknown data type: {type(received_data)} from {addr}")
        send_response(server_socket, "Unknown data type received!", addr)

# Usage example:
server_socket = start_server()
server_loop(server_socket)
