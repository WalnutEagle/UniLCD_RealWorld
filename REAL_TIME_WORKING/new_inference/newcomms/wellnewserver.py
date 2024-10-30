import socket
import pickle
import torch  # For PyTorch tensor handling
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
HOST = '0.0.0.0'  
PORT = 8083  
TIMEOUT = 15  # Timeout in seconds

# Function to start the server
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((HOST, PORT))
    server_socket.settimeout(TIMEOUT)  # Set timeout for the socket
    logging.info(f"Server listening on {HOST}:{PORT}...")
    return server_socket

# Function to receive data in chunks with timeout handling
def receive_data(server_socket):
    try:
        data_length, addr = server_socket.recvfrom(4)  # Receive length of data
        logging.info(f"Receiving data from {addr}...")
        data_length = int.from_bytes(data_length, 'big')
        data = b""
        while len(data) < data_length:
            packet, _ = server_socket.recvfrom(4096)  # Receive data in chunks
            data += packet
        received_data = pickle.loads(data)
        logging.info(f"Data received: {received_data} from {addr}")
        return received_data, addr
    except socket.timeout:
        logging.warning("Timeout occurred while receiving data.")
    except Exception as e:
        logging.error(f"Error receiving data: {e}")
    return None, None

# Function to send data with length prefix and timeout handling
def send_response(server_socket, response, addr):
    try:
        data = pickle.dumps(response)
        data_length = len(data)

        # Send the length of the data first
        server_socket.sendto(data_length.to_bytes(4, 'big'), addr)

        # Split data into smaller chunks if it's too large
        chunk_size = 1000  # Set an appropriate chunk size
        for i in range(0, data_length, chunk_size):
            server_socket.sendto(data[i:i + chunk_size], addr)  # Send each chunk
        logging.info("Response sent successfully.")
    except socket.timeout:
        logging.warning("Timeout occurred while sending data.")
    except Exception as e:
        logging.error(f"Error sending response: {e}")

# Main server loop function for processing data
def server_loop(server_socket):
    try:
        while True:
            received_data, addr = receive_data(server_socket)      
            if received_data is None:
                continue  # Skip the loop iteration if there was an error

            # Handle text or tensor data
            if isinstance(received_data, str):
                logging.info(f"Received text message: {received_data} from {addr}")
                send_response(server_socket, "Text received!", addr)
            elif isinstance(received_data, torch.Tensor):
                start_time = time.time()
                logging.info(f"Received PyTorch tensor data: \n{received_data} from {addr}")
                elapsed_time = (time.time() - start_time) * 1000
                logging.info(f"It took {elapsed_time:.2f} milliseconds.")
                
                tensor_data = torch.rand(1, 32, 150, 150)  # Example tensor response
                send_response(server_socket, tensor_data, addr)
            else:
                logging.warning(f"Received unknown data type: {type(received_data)} from {addr}")
                send_response(server_socket, "Unknown data type received!", addr)
    except Exception as e:
        logging.error(f"Server error: {e}")
    finally:
        server_socket.close()
        logging.info("Socket closed.")

# Usage example:
if __name__ == "__main__":
    server_socket = start_server()
    server_loop(server_socket)
