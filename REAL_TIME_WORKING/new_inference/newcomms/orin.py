from newserver import send_response, receive_data, start_server
import torch
socket = start_server()
data = torch.rand(1,4,150,130)
icom, addr = receive_data(socket)
while True:
    send_response(socket, data, addr)
    res, addr = receive_data(socket)
    print(res)
    if KeyboardInterrupt:
        break