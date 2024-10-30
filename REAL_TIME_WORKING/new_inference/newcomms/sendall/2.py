from orin import server_loop, start_server
import torch
server_socket = start_server()
data = torch.rand(1,32,150,150)
server_loop(server_socket, data)