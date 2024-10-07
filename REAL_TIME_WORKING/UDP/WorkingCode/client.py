import socket

class Client:
    def __init__(self, host='128.197.164.42', port=53):  # Replace with actual server IP
        self.client_socket = socket.socket()
        self.client_socket.connect((host, port))

    def send(self, message):
        self.client_socket.send(message.encode())
        print(f"Sent to server: {message}")

    def receive(self):
        data = self.client_socket.recv(1024).decode()
        print(f"Received from server: {data}")
        return data

    def close(self):
        self.client_socket.close()


def main():
    client = Client()  # Change to actual server IP

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
