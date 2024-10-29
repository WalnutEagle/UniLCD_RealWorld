'''from cloudsidemodel import CustomRegNetY002
from client import connect_to_server, send_data , receive_response
import torch
import numpy as np
import torch.nn as nn
import timm
import time
import glob
import os

def load_model(model_path):
    checkpoint = torch.load(model_path)  # Load the entire checkpoint
    model = CustomRegNetY002()  # Initialize your model
    # state_dict = checkpoint
    # Strip 'module.' from the keys if the model was saved with Data Parallelism
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}  # Remove 'module.' prefix

    model.load_state_dict(state_dict)  # Load only the model state dict
    model.eval()  # Set the model to evaluation mode
    return model

def inferr():
    data = receive_response(client_socket)
    data.to(device)
    with torch.no_grad():
        prediction = model(data)
    send_data(client_socket, prediction)

if __name__ == '__main__': 
    model_path = '/opt/app-root/src/UniLCD_RealWorld/REAL_TIME_WORKING/Models/ovrft/overfit8_900.pth'
    model = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    client_sock = connect_to_server()
    send_data(client_sock, 'a')
    try:
        while True:
            inferr()
    except KeyboardInterrupt:
        client_sock.close()'''

from cloudsidemodel import CustomRegNetY002
from client import connect_to_server, send_data, receive_response
import torch
import time

def load_model(model_path):
    checkpoint = torch.load(model_path)
    model = CustomRegNetY002()
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def inferr(client_socket, model, device):
    while True:
        # Wait for data from server
        data = receive_response(client_socket).to(device)
        with torch.no_grad():
            prediction = model(data)  # Run inference
        send_data(client_socket, prediction)  # Send back prediction or result

if __name__ == '__main__':
    model_path = '/opt/app-root/src/UniLCD_RealWorld/REAL_TIME_WORKING/Models/ovrft/overfit8_900.pth'
    model = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    
    client_socket = connect_to_server()  # Establish connection

    try:
        inferr(client_socket, model, device)  # Start inference loop
    except KeyboardInterrupt:
        print("Closing connection...")
        client_socket.close()
