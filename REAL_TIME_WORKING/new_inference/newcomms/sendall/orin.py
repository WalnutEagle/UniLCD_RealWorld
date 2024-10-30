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
    logging.info(f"Server listening on {HOST}:{PORT}...")
    return server_socket

# Function to receive data from a client
def receive_data(conn):
    try:
        data_length_bytes = conn.recv(4)  # Receive length of data
        if not data_length_bytes:
            return None
        
        data_length = int.from_bytes(data_length_bytes, 'big')
        logging.info(f"Receiving data of length: {data_length}...")
        
        data = b""
        while len(data) < data_length:
            packet = conn.recv(4096)  # Receive data in chunks
            if not packet:
                break
            data += packet
            
        received_data = pickle.loads(data)
        logging.info(f"Data received: {received_data} from client.")
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
def server_loop(server_socket):
    try:
        i=1
        while i<2:
            conn, addr = server_socket.accept()  # Accept a new connection
            logging.info(f"Connection from {addr}")

            # Send initial data to the client first
            tensor_data = torch.rand(1, 32, 150, 150)  # Example tensor to send
            send_response(conn, tensor_data)
            logging.info("Initial tensor data sent to client.")

            # Now receive data from the client
            received_data = receive_data(conn)
            print(receive_data)

            # if received_data is None:
            #     logging.warning("No data received, closing connection.")
            #     conn.close()
            #     continue  # Skip to the next iteration if there was an error

            # # Handle received data
            # if isinstance(received_data, str):
            #     logging.info(f"Received text message: {received_data} from {addr}")
            #     send_response(conn, "Text received!")
            # elif isinstance(received_data, torch.Tensor):
            #     logging.info(f"Received PyTorch tensor data: \n{received_data} from {addr}")
            #     tensor_response = torch.rand(1, 32, 150, 150)  # Another example tensor response
            #     send_response(conn, tensor_response)
            # else:
            #     logging.warning(f"Received unknown data type: {type(received_data)} from {addr}")
            #     send_response(conn, "Unknown data type received!")
            i+=2

            conn.close()  # Close the connection after handling
    except Exception as e:
        logging.error(f"Server error: {e}")
    finally:
        server_socket.close()
        logging.info("Socket closed.")

# Usage example:
if __name__ == "__main__":
    server_socket = start_server()
    server_loop(server_socket)
