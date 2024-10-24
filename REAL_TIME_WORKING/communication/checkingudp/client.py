import socket
import pickle
import torch
import time

HOST = '128.197.164.42'
PORT = 8083

def connect_to_server():
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_data(client_socket, data):
    serialized_data = pickle.dumps(data)
    data_length = len(serialized_data)
    client_socket.sendto(data_length.to_bytes(4, 'big'), (HOST, PORT))
    client_socket.sendto(serialized_data, (HOST, PORT))

def receive_response(client_socket):
    data_length, _ = client_socket.recvfrom(4)
    data_length = int.from_bytes(data_length, 'big')
    data = b""
    while len(data) < data_length:
        packet, _ = client_socket.recvfrom(4096)  # Increased buffer size for fewer packets
        data += packet
    return pickle.loads(data)

def client_loop(client_socket):
    while True:
        choice = input("Enter 't' for text, 'n' for tensor, 'q' to quit: ")
        if choice == 't':
            text_message = input("Enter your text message: ")
            send_data(client_socket, text_message)
            print("Server response:", receive_response(client_socket))
        elif choice == 'n':
            tensor_data = torch.rand(2, 2)
            start = time.time()
            print(f"Sending PyTorch tensor: \n{tensor_data}")
            send_data(client_socket, tensor_data)
            print(f"Data sent in {(time.time()-start)*1000:.2f} ms")
            print("Server response:", receive_response(client_socket))
        elif choice == 'q':
            print("Closing connection...")
            break
    client_socket.close()

if __name__ == "__main__":
    client_socket = connect_to_server()
    client_loop(client_socket)
