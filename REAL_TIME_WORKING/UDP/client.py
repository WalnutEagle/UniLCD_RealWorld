import socket
import threading
import queue
import numpy as np

# Shared queue to hold received messages
message_queue = queue.Queue()

def start_client(server_ip, server_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def receive_messages():
        while True:
            data, _ = sock.recvfrom(1024 * 1024)  # Larger buffer size for file transfers
            print(f"Received message: {data.decode()}")
            message_queue.put(data.decode())

    threading.Thread(target=receive_messages, daemon=True).start()
    print(f"Client ready to send messages to {server_ip}:{server_port}")

    return sock

def send_message(sock, message, server_ip, server_port):
    sock.sendto(message.encode(), (server_ip, server_port))

def send_file(sock, file_path, server_ip, server_port):
    with open(file_path, 'rb') as file:
        file_data = file.read()
        sock.sendto(file_data, (server_ip, server_port))

def send_tensor(sock, tensor, server_ip, server_port):
    tensor_bytes = tensor.tobytes()
    sock.sendto(tensor_bytes, (server_ip, server_port))

def get_latest_message():
    if not message_queue.empty():
        return message_queue.get()
    return None

# Usage
if __name__ == "__main__":
    server_ip = 'actual_server_ip_here'  # Replace with the actual server IP
    client_sock = start_client(server_ip, 12345)

    # Example of sending a text message
    send_message(client_sock, "Hello from client!", server_ip, 12345)

    # Example of sending a file
    send_file(client_sock, 'path/to/your/file.txt', server_ip, 12345)

    # Example of sending a NumPy tensor
    tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    send_tensor(client_sock, tensor, server_ip, 12345)

    # Example loop to process received messages
    while True:
        latest_message = get_latest_message()
        if latest_message:
            print(f"Processed received message: {latest_message}")
