from newclient import receive_response, send_data, connect_to_server
import torch

socket = connect_to_server()
send_data(socket, '', timeout=5)
while True:
    tensord = torch.rand(2, 2)
    send_data(socket, tensord)
    response = receive_response(socket)
    if KeyboardInterrupt:
        break