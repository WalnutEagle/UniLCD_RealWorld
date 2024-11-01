import socket 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 8083))
while True:
    msg , addr = sock.recvfrom(1024)
    print(msg)