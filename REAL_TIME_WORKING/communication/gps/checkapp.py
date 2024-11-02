import socket

def start_udp_server():
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Bind the socket to the address and port
    server_address = ('0.0.0.0', 8083)  # Listening on all interfaces
    sock.bind(server_address)

    print(f"Listening for UDP packets on {server_address[0]}:{server_address[1]}...")

    while True:
        # Wait for a message
        data, address = sock.recvfrom(4096)  # Buffer size is 4096 bytes
        print(f"Received {data} from {address}")

if __name__ == "__main__":
    start_udp_server()