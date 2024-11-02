import socket

# Define the server address and port
server_ip = '10.239.28.208'
server_port = 8080

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # Connect to the server
    client_socket.connect((server_ip, server_port))
    print(f"Connected to server {server_ip} on port {server_port}")

    # Receive data from the server
    while True:
        # Here, we receive up to 1024 bytes of data
        data = client_socket.recv(1024)
        if not data:
            break  # Exit if there's no more data
        print(f"Received: {data.decode('utf-8')}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the socket
    client_socket.close()
    print("Connection closed.")
