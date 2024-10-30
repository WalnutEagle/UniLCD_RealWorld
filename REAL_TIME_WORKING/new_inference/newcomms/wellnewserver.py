import socket
import pickle
import torch  
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
HOST = '0.0.0.0'  
PORT = 8083  
TIMEOUT = 10

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((HOST, PORT))
    server_socket.settimeout(TIMEOUT)
    logging.info(f"Server listening on {HOST}:{PORT}...")
    return server_socket

def receive_data(server_socket):
    try:
        data_length, addr = server_socket.recvfrom(4)  
        logging.info(f"Receiving data from {addr}...")
        data_length = int.from_bytes(data_length, 'big')
        data = b""
        while len(data) < data_length:
            packet, _ = server_socket.recvfrom(4096)  
            data += packet
        received_data = pickle.loads(data)
        logging.info(f"Data received: {received_data} from {addr}")
        return received_data, addr
    except socket.timeout:
        logging.warning("Timeout occurred while receiving data.")
    except Exception as e:
        logging.error(f"Error receiving data: {e}")
    return None, None

def send_response(server_socket, response, addr):
    try:
        data = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        data_length = len(data)
        server_socket.sendto(data_length.to_bytes(4, 'big'), addr)
        chunk_size = 1000 
        for i in range(0, data_length, chunk_size):
            server_socket.sendto(data[i:i + chunk_size], addr) 
        logging.info("Response sent successfully.")
    except socket.timeout:
        logging.warning("Timeout occurred while sending data.")
    except Exception as e:
        logging.error(f"Error sending response: {e}")

def server_loop(server_socket):
    try:
        while True:
            received_data, addr = receive_data(server_socket)      
            if received_data is None:
                continue 

            if isinstance(received_data, str):
                logging.info(f"Received text message: {received_data} from {addr}")
                send_response(server_socket, "Text received!", addr)
            elif isinstance(received_data, torch.Tensor):
                start_time = time.time()
                logging.info(f"Received PyTorch tensor data: \n{received_data} from {addr}")
                elapsed_time = (time.time() - start_time) * 1000
                logging.info(f"It took {elapsed_time:.2f} milliseconds.")
                
                tensor_data = torch.rand(1, 32, 150, 150)
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




