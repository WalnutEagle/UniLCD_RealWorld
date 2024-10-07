import socket

class UDPServer:
    def __init__(self, host='0.0.0.0', port=8083):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.bind((host, port))
        print(f"UDP Server is waiting for a connection on {host}:{port}...")
        self.last_received_message = None

    def send(self, message, address):
        self.server_socket.sendto(message.encode(), address)
        print(f"Sent to {address}: {message}")

    def receive(self):
        data, address = self.server_socket.recvfrom(1024)
        self.last_received_message = data.decode()
        print(f"Received from {address}: {self.last_received_message}")
        return self.last_received_message, address

    def get_last_received_message(self):
        return self.last_received_message

    def close(self):
        self.server_socket.close()


def main():
    server = UDPServer()
    
    try:
        while True:
            message, address = server.receive()  # Wait for messages
            server.send("Hello from UDP Server!", address)  # Respond to the client
    except KeyboardInterrupt:
        print("Server shutting down.")
    finally:
        server.close()


if __name__ == '__main__':
    main()
