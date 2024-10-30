'''from udpc import receive_response, send_data, connect_to_server
import torch

if __name__=='__main__':
    try:
        while True:
            socket_1 = connect_to_server()
            tensord = torch.rand(2, 2)
            response = receive_response(socket_1)
            print(response)
            send_data(socket_1, tensord)
    except KeyboardInterrupt:
        print('Bye')
        socket_1.close()'''

import socket
import pickle
import torch  # For PyTorch tensor handling
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Server configuration
HOST = '128.197.164.42'  # Server's IP address
PORT = 8083              # Server's port

# Function to connect to the server
def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(20)
    client_socket.connect((HOST, PORT))  # Connect to the server
    return client_socket

# Function to send data (text or tensor) to the server
def send_data(client_socket, data):
    serialized_data = pickle.dumps(data)
    data_length = len(serialized_data)
    start_time = time.time()
    
    try:
        client_socket.sendall(data_length.to_bytes(4, 'big'))  # Send length
        client_socket.sendall(serialized_data)  # Send data
        
        elapsed_time = time.time() - start_time
        logging.info("Data sent successfully in %.2f seconds.", elapsed_time)
        return True
    except Exception as e:
        logging.error(f"Error sending data: {e}")
        return False

# Function to receive response from the server
def receive_response(client_socket):
    try:
        data_length_bytes = client_socket.recv(4)  # Receive length of the data
        if not data_length_bytes:
            return None

        data_length = int.from_bytes(data_length_bytes, 'big')
        logging.info(f"Receiving data of length: {data_length}...")

        data = b""
        while len(data) < data_length:
            packet = client_socket.recv(4096)  # Receive data in chunks
            if not packet:
                break
            data += packet

        if len(data) != data_length:
            logging.error(f"Received {len(data)} bytes, expected {data_length} bytes.")
            return None
        
        response = pickle.loads(data)
        logging.info("Response received successfully.")
        return response
    except Exception as e:
        logging.error(f"Error receiving response: {e}")
        return None

# Main client communication loop
def client_loop():
    i=1
    while i<2:
        client_socket = connect_to_server()

        # Receive data from the server first
        initial_data = receive_response(client_socket)
        if initial_data is not None:
            logging.info(f"Initial data received from server: {initial_data}")

            # Send data back to the server
            tensor_data = torch.rand(2, 2)
            send_data(client_socket, tensor_data)  # Example tensor data
            logging.info(f"Sending PyTorch tensor: \n{tensor_data}")
            # if send_data(client_socket, tensor_data):
            #     response = receive_response(client_socket)
            #     if response is not None:
            #         logging.info("Server response: %s", response)

        else:
            logging.warning("No initial data received from server, closing connection.")
            break
        i+=2

    client_socket.close()
    logging.info("Socket closed.")

# Usage example:
if __name__ == "__main__":
    client_loop()
