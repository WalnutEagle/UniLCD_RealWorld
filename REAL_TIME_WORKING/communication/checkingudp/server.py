import socket
import torch
import io
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
    return data, addr

def send_response(server_socket, response, addr):
    if isinstance(response, torch.Tensor):
        response = serialize_tensor(response)  # Serialize tensor
    else:
        response = response.encode('utf-8')  # Ensure response is bytes
    
    data_length = len(response)
    server_socket.sendto(data_length.to_bytes(4, 'big'), addr)
    server_socket.sendto(response, addr)

def serialize_tensor(tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()

def deserialize_tensor(data):
    buffer = io.BytesIO(data)
    return torch.load(buffer)

def server_loop(server_socket):
    while True:
        data, addr = receive_data(server_socket)
        
        if data.startswith(b'text:'):
            message = data[5:].decode('utf-8')
            print(f"Received text message: {message} from {addr}")
            send_response(server_socket, "Text received!", addr)
        else:
            try:
                tensor_data = deserialize_tensor(data)
                print(f"Received tensor data: \n{tensor_data} from {addr}")
                response_tensor = torch.rand(1, 4, 150, 130)  # Create a tensor
                send_response(server_socket, response_tensor, addr)
            except Exception as e:
                print(f"Error unpacking data: {e}")
                send_response(server_socket, "Unknown data type received!", addr)

if __name__ == "__main__":
    server_socket = start_server()
    server_loop(server_socket)
