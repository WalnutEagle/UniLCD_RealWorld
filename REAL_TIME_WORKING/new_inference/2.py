import socket
import pickle
import torch  # For PyTorch tensor handling
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
HOST = '0.0.0.0'  
PORT = 8083  
TIMEOUT = 20  # Timeout in seconds

# Function to start the server
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)  # Listen for incoming connections
    server_socket.settimeout(TIMEOUT)  # Set timeout for the socket
    # logging.info(f"Server listening on {HOST}:{PORT}...")
    return server_socket

# Function to receive data from a client
def receive_data(conn):
    try:
        data_length_bytes = conn.recv(4)  # Receive length of data
        if not data_length_bytes:
            return None
        
        data_length = int.from_bytes(data_length_bytes, 'big')
        # logging.info(f"Receiving data of length: {data_length}...")
        
        data = b""
        while len(data) < data_length:
            packet = conn.recv(4096)  # Receive data in chunks
            if not packet:
                break
            data += packet
            
        received_data = pickle.loads(data)
        # logging.info(f"Data received: {received_data} from client.")
        return received_data
    except Exception as e:
        logging.error(f"Error receiving data: {e}")
    return None

# Function to send data back to the client
def send_response(conn, response):
    try:
        data = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        data_length = len(data)

        # Send the length of the data first
        conn.sendall(data_length.to_bytes(4, 'big'))  # Send length as bytes

        # Send the actual data
        conn.sendall(data)
        logging.info("Response sent successfully.")
    except Exception as e:
        logging.error(f"Error sending response: {e}")

# Main server loop function for processing data
def server_loop(server_socket, conn, addr, data):
    try:
        while True:
              # Accept a new connection
            logging.info(f"Connection from {addr}")
            start = time.time()
            # tensor_data = torch.rand(1, 32, 150, 150) 
            send_response(conn, data)
            # logging.info("Initial tensor data sent to client.")

            # Now receive data from the client
            received_data = receive_data(conn)
            print(f"Total Time is equal to {(time.time()-start)*1000}Miliseconds.")
            print(received_data)

            # conn.close()
    except Exception as e:
        logging.error(f"Server error: {e}")
    finally:
        return received_data 
        # server_socket.close()
        logging.info("Socket closed.")

# Usage example:
if __name__ == "__main__":
    tensor_data = torch.rand(1, 32, 150, 150)
    server_socket = start_server()
    conn, addr = server_socket.accept()
    server_loop(server_socket, conn, addr, tensor_data)
