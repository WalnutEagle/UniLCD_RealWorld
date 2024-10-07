import socket

class UDPClient:
    def __init__(self, host='128.197.164.42', port=53):  # Connect to the specified server IP and Port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = (host, port)

    def send(self, message):
        self.client_socket.sendto(message.encode(), self.server_address)  # Send to the server
        print(f"Sent to server: {message}")

    def receive(self):
        data, _ = self.client_socket.recvfrom(1024)
        message = data.decode()
        print(f"Received from server: {message}")
        return message

    def close(self):
        self.client_socket.close()


def main():
    client = UDPClient()  # Connect to the specified server

    try:
        while True:
            message = input("Enter message to send (or 'exit' to quit): ")
            if message.lower() == 'exit':
                break
            client.send(message)
            response = client.receive()  # Get response from server
    finally:
        client.close()


if __name__ == '__main__':
    main()
