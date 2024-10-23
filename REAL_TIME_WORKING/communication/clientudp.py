import socket
import pickle
import torch  # For PyTorch tensor handling
import time
import struct

# Define server address and port
HOST = '128.197.164.42'  # Replace with actual server IP for remote
PORT = 8083              # Server port

# Set up the client UDP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Function to send data (text or tensor) to server
def send_data(data):
    client_socket.sendto(pickle.dumps(data), (HOST, PORT))

# Function to receive response from server
def receive_response():
    response, _ = client_socket.recvfrom(4096)
    return pickle.loads(response)

# Function to receive large data
def receive_large_data():
    received_data = b""
    total_size = None

    while True:
        chunk, _ = client_socket.recvfrom(4096)
        print(f"Received chunk of size: {len(chunk)}")  # Debugging line

        if len(chunk) == 0:  # If an empty chunk is received, end reception
            break

        # Check if we received the header indicating total size
        if total_size is None:
            # Assume the first 4 bytes contain the total size as an integer
            if len(chunk) >= 4:
                total_size = struct.unpack('!I', chunk[:4])[0]  # Big-endian unsigned int
                received_data += chunk[4:]  # Add the rest of the chunk
            else:
                # If the chunk is smaller than 4 bytes, it's not a valid header
                continue
        else:
            received_data += chunk

        # If we've received enough data, we can break out
        if len(received_data) >= total_size:
            break

    # Now attempt to unpickle the data after ensuring all chunks are received
    if received_data:
        try:
            return pickle.loads(received_data)
        except pickle.UnpicklingError as e:
            raise ValueError(f"Failed to unpickle data: {e}")
    else:
        raise ValueError("No data received before end of transmission")





# Example loop to send data
def client_loop():
    while True:
        choice = input("Enter 't' for text, 'p' for PyTorch tensor, 'q' to quit: ")

        if choice == 't':
            text_message = input("Enter your text message: ")
            send_data(text_message)
            print("Server response:", receive_response())
            
        elif choice == 'p':
            start = time.time()
            tensor_data = torch.rand(2, 2)  # Example PyTorch tensor
            print(f"Sending PyTorch tensor: \n{tensor_data}")
            send_data(tensor_data)  # Send as bytes
            print(f"Data sent in {time.time()-start} seconds")
            data = receive_large_data()
            print("Received tensor from server:\n", data)
            
        elif choice == 'q':
            print("Closing connection...")
            break

    client_socket.close()

# Usage:
client_loop()
