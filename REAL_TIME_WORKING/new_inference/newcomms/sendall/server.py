import socket
import pickle
import torch

HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 8083

# Create a sample tensor
tensor = torch.randn(1, 32, 150, 150)

# Serialization Verification
test_serialized = pickle.dumps(tensor, protocol=pickle.HIGHEST_PROTOCOL)
test_tensor = pickle.loads(test_serialized)
assert torch.equal(tensor, test_tensor), "Tensor serialization/deserialization failed."

# Set up server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("Server is listening on {}:{}".format(HOST, PORT))

conn, addr = server_socket.accept()
print(f"Connection from {addr}")

# Serialize and send the tensor
serialized_tensor = pickle.dumps(tensor, protocol=pickle.HIGHEST_PROTOCOL)
conn.sendall(serialized_tensor)
print("Tensor sent successfully.")
conn.close()
