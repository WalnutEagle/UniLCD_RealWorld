from orin import server_loop, start_server
server_socket = start_server()
server_loop(server_socket)