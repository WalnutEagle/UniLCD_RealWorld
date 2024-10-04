'''import socket
import threading
import queue

# Shared queue to hold received messages
message_queue = queue.Queue()

def start_server(host='0.0.0.0', port=12345):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))

    def receive_messages():
        while True:
            data, addr = sock.recvfrom(1024)  # Buffer size
            print(f"Received message: {data.decode()} from {addr}")
            # Put the received message in the queue
            message_queue.put(data.decode())

    threading.Thread(target=receive_messages, daemon=True).start()
    print(f"Server listening on {host}:{port}")

def send_message(sock, message, target_ip, target_port):
    sock.sendto(message.encode(), (target_ip, target_port))

def get_latest_message():
    """Retrieve the latest message from the queue if available."""
    if not message_queue.empty():
        return message_queue.get()
    return None

# Start the server
if __name__ == "__main__":
    start_server()

    # Example loop to process received messages
    while True:
        latest_message = get_latest_message()
        if latest_message:
            print(f"Processed received message: {latest_message}")
            # You can add your logic here to do further operations on the message
'''

import socket
import threading
import queue
import os

# Shared queue to hold received messages
message_queue = queue.Queue()

def start_server(host='0.0.0.0', port=12345):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))

    def receive_messages():
        while True:
            # Increase buffer size for larger data
            data, addr = sock.recvfrom(65536)  # 64KB buffer size
            print(f"Received data from {addr}")

            # Handle received data
            handle_received_data(data, addr)

    threading.Thread(target=receive_messages, daemon=True).start()
    print(f"Server listening on {host}:{port}")

def handle_received_data(data, addr):
    # You can customize this function based on the expected data format
    # For example, if you expect images, you could save them to a file:
    
    # Save data to a file (example for image)
    with open(f"received_data_from_{addr[0]}.bin", 'wb') as f:
        f.write(data)
    print(f"Data saved as received_data_from_{addr[0]}.bin")

    # If it's a tensor or another type, handle accordingly
    # Example: Convert to a numpy array if you know the shape beforehand

def send_message(sock, message, target_ip, target_port):
    sock.sendto(message.encode(), (target_ip, target_port))

def get_latest_message():
    """Retrieve the latest message from the queue if available."""
    if not message_queue.empty():
        return message_queue.get()
    return None

# Start the server
if __name__ == "__main__":
    start_server()

    # Example loop to process received messages
    while True:
        latest_message = get_latest_message()
        if latest_message:
            print(f"Processed received message: {latest_message}")
            # You can add your logic here to do further operations on the message
