from newserver import send_response, receive_data, start_server
import torch
socket = start_server()
data = torch.rand(1,4,150,130)
send_response