import socket
import msgpack
import torch
import io
import time

HOST = '128.197.164.42'
PORT = 8083

def connect_to_server():
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def serialize_tensor(tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()

def deserialize_tensor(data):
    buffer = io.BytesIO(data)
    return torch.load(buffer)

def send_data(client_socket, data):
    if isinstance(data, torch.Tensor):
        data = serialize_tensor(data)  # Serialize tensor
    else:
        data = msgpack.packb(data)  # Serialize other data types
    data_length = len(data)
    client_socket.sendto(data_length.to_bytes(4, 'big'), (HOST, PORT))
    client_socket.sendto(data, (HOST, PORT))

def receive_response(client_socket):
    try:
        data_length, _ = client_socket.recvfrom(4)
        data_length = int.from_bytes(data_length, 'big')
        data = b""
        while len(data) < data_length:
            packet, _ = client_socket.recvfrom(4096)
            data += packet
        
        return msgpack.unpackb(data) if data_length < 4096 else deserialize_tensor(data)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def client_loop(client_socket):
    while True:
        choice = input("Enter 't' for text, 'n' for tensor, 'q' to quit: ")
        if choice == 't':
            text_message = input("Enter your text message: ")
            send_data(client_socket, text_message)
            response = receive_response(client_socket)
            print("Server response:", response if response else "No valid response received.")
        elif choice == 'n':
            tensor_data = torch.rand(2, 2)  # Create a tensor
            start = time.time()
            print(f"Sending PyTorch tensor: \n{tensor_data}")
            send_data(client_socket, tensor_data)  # Send tensor directly
            print(f"Data sent in {(time.time()-start)*1000:.2f} ms")
            response = receive_response(client_socket)
            print("Server response:", response if response else "No valid response received.")
        elif choice == 'q':
            print("Closing connection...")
            break
    client_socket.close()

if __name__ == "__main__":
    client_socket = connect_to_server()
    client_loop(client_socket)
