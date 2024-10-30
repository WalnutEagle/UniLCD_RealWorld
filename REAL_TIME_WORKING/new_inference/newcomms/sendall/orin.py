from udps import send_response, receive_data, start_server
import torch


if __name__=='__main__':
    # Start the server
    server_2_soc = start_server()  # Renamed from 'socket' to 'server_2_soc'
    data = torch.rand(1, 32, 150, 150)  # Sample tensor data

    # print(addr)
    try:
        while True:
            conn, addr = server_2_soc.accept()
            send_response(server_2_soc, data)
            res = receive_data(server_2_soc)
            print(res)
            
    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        server_2_soc.close()
        print("Socket closed.")
