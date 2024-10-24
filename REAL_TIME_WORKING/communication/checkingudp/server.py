import socket
import msgpack
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
    
    try:
        # Check if the received data is a serialized tensor
        if data_length > 4096:  # Arbitrary threshold, adjust as needed
            return deserialize_tensor(data), addr
        else:
            return msgpack.unpackb(data), addr
    except Exception as e:
        print(f"Error unpacking data: {e}")
        return None, addr

def send_response(server_socket, response, addr):
    if isinstance(response, torch.Tensor):
        response = serialize_tensor(response)  # Serialize tensor
    else:
        response = msgpack.packb(response)  # Serialize other data types
    
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
        received_data, addr = receive_data(server_socket)
        if isinstance(received_data, str):
            print(f"Received text message: {received_data} from {addr}")
            send_response(server_socket, "Text received!", addr)
        elif isinstance(received_data, torch.Tensor):
            print(f"Received tensor data: \n{received_data} from {addr}")
            tensor_data = torch.rand(1, 4, 150, 130)  # Create a tensor
            t1 = time.time()
            send_response(server_socket, tensor_data, addr)  # Send tensor back
            print(f"Tensor sent in {(time.time()-t1)*1000:.2f} ms")
        else:
            print(f"Received unknown data type: {type(received_data)} from {addr}")
            send_response(server_socket, "Unknown data type received!", addr)

if __name__ == "__main__":
    server_socket = start_server()
    server_loop(server_socket)
