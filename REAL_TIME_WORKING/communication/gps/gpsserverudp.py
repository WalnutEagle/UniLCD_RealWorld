import socket

# Define the server address and port
server_ip = '10.239.28.208'
server_port = 8080

# Create a UDP socket
udp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the server address and port
udp_client_socket.bind((server_ip, server_port))

print(f"Listening for data on {server_ip}:{server_port}")

try:
    while True:
        # Receive data from the server
        data, addr = udp_client_socket.recvfrom(4096)  # Buffer size of 1024 bytes
        print(f"Received: {data.decode('utf-8')} from {addr}")

except KeyboardInterrupt:
    print("Stopped listening for data.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the socket
    udp_client_socket.close()
    print("Connection closed.")
