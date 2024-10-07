import socket

class UDPClient:
    def __init__(self, host='128.197.164.42', port=53):
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

    def get_last_response(self):
        return self.receive()  # Call receive to get the last response

    def close(self):
        self.client_socket.close()


def main():
    client = UDPClient()  # Connect to the specified server

    try:
        while True:
            message = input("Enter message to send (or 'exit' to quit): ")
            if message.lower() == 'exit':
                break
            client.send(message)  # Send the message
            response = client.get_last_response()  # Get response from server
            # You can save the response to a variable here
            print(f"Processed response: {response}")  # Do further processing if needed
    finally:
        client.close()


if __name__ == '__main__':
    main()
