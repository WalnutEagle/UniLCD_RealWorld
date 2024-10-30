from nerc import client_loop, receive_response, send_data, connect_to_server
import torch
client_socket = connect_to_server()
initial_data = receive_response(client_socket)
if initial_data is not None:


    # Send data back to the server
    tensor_data = torch.rand(2, 2)
    send_data(client_socket, tensor_data)
