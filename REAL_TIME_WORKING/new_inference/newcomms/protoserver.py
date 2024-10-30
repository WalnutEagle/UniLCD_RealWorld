import socket
import torch  # For PyTorch tensor handling
import time
import logging
import message_pb2  # Import the generated protobuf classes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
HOST = '0.0.0.0'  
PORT = 8083  
TIMEOUT = 20  # Timeout in seconds

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
        
        # Deserialize protobuf message
        message = message_pb2.Message()
        message.ParseFromString(data)
        
        logging.info(f"Data received: {message} from {addr}")
        return message, addr
    except socket.timeout:
        logging.warning("Timeout occurred while receiving data.")
    except Exception as e:
        logging.error(f"Error receiving data: {e}")
    return None, None

# Function to send data with length prefix and timeout handling
def send_response(server_socket, response, addr):
    try:
        data = response.SerializeToString()
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
            message, addr = receive_data(server_socket)      
            if message is None:
                continue  # Skip the loop iteration if there was an error

            # Handle text or tensor data
            if message.HasField('text_message'):
                text_received = message.text_message.text
                logging.info(f"Received text message: {text_received} from {addr}")
                response = message_pb2.Message()
                response.text_message.text = "Text received!"
                send_response(server_socket, response, addr)
            elif message.HasField('tensor_message'):
                tensor_data = torch.rand(1, 32, 150, 150)  # Example tensor response
                tensor_bytes = tensor_data.numpy().tobytes()  # Convert tensor to bytes
                response = message_pb2.Message()
                response.tensor_message.tensor_data = tensor_bytes
                logging.info(f"Received tensor data from {addr}. Sending tensor response.")
                send_response(server_socket, response, addr)
            else:
                logging.warning(f"Received unknown data type from {addr}")
                response = message_pb2.Message()
                response.text_message.text = "Unknown data type received!"
                send_response(server_socket, response, addr)
    except Exception as e:
        logging.error(f"Server error: {e}")
    finally:
        server_socket.close()
        logging.info("Socket closed.")

# Usage example:
if __name__ == "__main__":
    server_socket = start_server()
    server_loop(server_socket)