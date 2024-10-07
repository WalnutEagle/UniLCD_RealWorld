import socket

class Server:
    def __init__(self, host='192.168.0.177', port=8083):
        self.server_socket = socket.socket()
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print("Server is waiting for a connection...")
        self.conn, self.address = self.server_socket.accept()
        print(f"Connection from: {self.address}")
        self.last_received_message = None

    def send(self, message):
        self.conn.send(message.encode())
        print(f"Sent to client: {message}")

    def receive(self):
        data = self.conn.recv(1024).decode()
        if data:
            self.last_received_message = data
            print(f"Received from client: {data}")
        return self.last_received_message

    def close(self):
        self.conn.close()
        self.server_socket.close()


def main():
    server = Server()
    
    try:
        while True:
            server.receive()  # Wait for messages
    except KeyboardInterrupt:
        print("Server shutting down.")
    finally:
        server.close()


if __name__ == '__main__':
    main()
