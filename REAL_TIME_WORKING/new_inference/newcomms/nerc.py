from sendall.udpc import receive_response, send_data, connect_to_server
import torch


try:
    while True:
        socket_1 = connect_to_server()
        tensord = torch.rand(2, 2)
        response = receive_response(socket_1)
        print(response)
        send_data(socket_1, tensord)
except KeyboardInterrupt:
    print('Bye')
    socket_1.close()