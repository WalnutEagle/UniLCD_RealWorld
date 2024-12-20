import socket
import pickle
import torch  
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HOST = '128.197.164.42'  
PORT = 8083             

def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(20) 
    return client_socket

def send_data(client_socket, data, timeout=20):
    serialized_data = pickle.dumps(data)
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
    
def receive_response(client_socket):
    try:
        data_length, addr = client_socket.recvfrom(4)
        data_length = int.from_bytes(data_length, 'big')
        logging.info(f"Reciving data of lenght:{data_length} from {addr}")
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
        response = pickle.loads(data)
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
        choice = 'n'

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
                t1 = time.time()
                response = receive_response(client_socket)
                if response is not None:
                    logging.info("Server response: %s", response)
                logging.info(f"Received data in: {(time.time()-t1)*1000:.2f} milliseconds.")

        elif choice == 'q':
            logging.info("Closing connection...")
            break

    client_socket.close()
    logging.info("Socket closed.")

if __name__ == "__main__":
    client_socket = connect_to_server()
    client_loop(client_socket)




