from newserver import send_response, receive_data, start_server
import torch

# Start the server
server_2_soc = start_server()  # Renamed from 'socket' to 'server_2_soc'
data = torch.rand(1, 4, 150, 140)  # Sample tensor data
received_data, addr = receive_data(server_2_soc)
print(addr)
try:
    while True:
        send_response(server_2_soc, data, addr)
        res, addr = receive_data(server_2_soc)
        print(res)
        
except KeyboardInterrupt:
    print("Server shutting down...")
finally:
    server_2_soc.close()
    print("Socket closed.")
