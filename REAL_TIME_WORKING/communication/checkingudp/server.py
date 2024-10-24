import socket
import msgpack
import torch
import time

HOST = '0.0.0.0'
PORT = 8083

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((HOST, PORT))
    print(f"Server listening on {HOST}:{PORT}...")
    return server_socket

def receive_data(server_socket):
    data_length, addr = server_socket.recvfrom(4)
    data_length = int.from_bytes(data_length, 'big')
    data = b""
    while len(data) < data_length:
        packet, _ = server_socket.recvfrom(4096)
        data += packet
    return msgpack.unpackb(data), addr  # Use MessagePack for deserialization

def send_response(server_socket, response, addr):
    data = msgpack.packb(response)  # Use MessagePack for serialization
    data_length = len(data)
    server_socket.sendto(data_length.to_bytes(4, 'big'), addr)
    server_socket.sendto(data, addr)

def server_loop(server_socket):
    while True:
        received_data, addr = receive_data(server_socket)
        if isinstance(received_data, str):
            print(f"Received text message: {received_data} from {addr}")
            send_response(server_socket, "Text received!", addr)
        elif isinstance(received_data, list):  # Expecting list from tensor conversion
            print(f"Received tensor data: {received_data} from {addr}")
            tensor_data = torch.rand(1, 4, 150, 130).tolist()  # Convert tensor to list
            t1 = time.time()
            send_response(server_socket, tensor_data, addr)
            print(f"Tensor sent in {(time.time()-t1)*1000:.2f} ms")
        else:
            print(f"Received unknown data type: {type(received_data)} from {addr}")
            send_response(server_socket, "Unknown data type received!", addr)

if __name__ == "__main__":
    server_socket = start_server()
    server_loop(server_socket)
