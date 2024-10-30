from newclient import receive_response, send_data, connect_to_server
import torch

socket_1 = connect_to_server()
send_data(socket_1, '', timeout=15)
try:
    while True:
        tensord = torch.rand(2, 2)

        response = receive_response(socket_1)
        send_data(socket_1, tensord)
except KeyboardInterrupt:
    print('Bye')
    socket_1.close()