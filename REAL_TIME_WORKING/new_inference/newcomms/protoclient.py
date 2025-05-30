import socket
import torch  # For PyTorch tensor handling
import time
import logging
import message_pb2  # Import the generated protobuf classes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Server configuration
HOST = '128.197.164.42'  # Server's IP address
PORT = 8083              # Server's port

# Function to connect to the server
def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(20)
    return client_socket

# Function to send data (text or tensor) to the server
def send_data(client_socket, data, timeout=20):
    message = message_pb2.Message()

    # Determine if the data is text or tensor
    if isinstance(data, str):
        message.text_message.text = data
    elif isinstance(data, torch.Tensor):
        message.tensor_message.tensor_data = data.numpy().tobytes()  # Convert tensor to bytes

    serialized_data = message.SerializeToString()
    data_length = len(serialized_data)
    start_time = time.time()
    
    try:
        client_socket.sendto(data_length.to_bytes(4, 'big'), (HOST, PORT))
        client_socket.sendto(serialized_data, (HOST, PORT))
        
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            logging.error("Sending data timed out.")
            return False

        logging.info("Data sent successfully in %.2f seconds.", elapsed_time)
        return True
    except Exception as e:
        logging.error(f"Error sending data: {e}")
        return False

# Function to receive response from the server in chunks
def receive_response(client_socket):
    try:
        data_length, addr = client_socket.recvfrom(4)
        data_length = int.from_bytes(data_length, 'big')
        logging.info(f"Receiving data of length: {data_length} from {addr}")
        data = b""
        while len(data) < data_length:
            try:
                packet, _ = client_socket.recvfrom(1000)
                data += packet
            except socket.timeout:
                logging.info("Timed out while waiting for packet.")

        if len(data) != data_length:
            logging.error(f"Received {len(data)} bytes, expected {data_length} bytes.")
            return None

        # Deserialize the protobuf message
        response = message_pb2.Message()
        response.ParseFromString(data)
        logging.info("Response received successfully.")
        return response
    except socket.timeout:
        logging.error("Receiving data timed out.")
        return None
    except Exception as e:
        logging.error(f"Error receiving response: {e}")
        return None

# Main client communication loop
def client_loop(client_socket):
    while True:
        choice = input("Enter 't' for text, 'n' for tensor, 'q' to quit: ")

        if choice == 't':
            text_message = input("Enter your text message: ")
            if send_data(client_socket, text_message):
                response = receive_response(client_socket)
                if response is not None:
                    logging.info("Server response: %s", response)

        elif choice == 'n':
            tensor_data = torch.rand(2, 2)
            start = time.time()
            logging.info(f"Sending PyTorch tensor: \n{tensor_data}")
            if send_data(client_socket, tensor_data):
                response = receive_response(client_socket)
                if response is not None:
                    logging.info("Server response: %s", response)
                logging.info(f"Received data in: {(time.time() - start) * 1000:.2f} milliseconds.")

        elif choice == 'q':
            logging.info("Closing connection...")
            break

    client_socket.close()
    logging.info("Socket closed.")

# Usage example:
if __name__ == "__main__":
    client_socket = connect_to_server()
    client_loop(client_socket)