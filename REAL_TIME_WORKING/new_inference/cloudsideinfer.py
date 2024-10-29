from cloudsidemodel import CustomRegNetY002
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

def inferr(device):
    send_data(client_sock, '')
    data = receive_response(client_sock)

    if data is not None:
        print('got it')
        print(data)
        data.to(device)
        with torch.no_grad():
            prediction = model(data)
        send_data(client_sock, prediction)



if __name__ == '__main__': 
    model_path = '/opt/app-root/src/UniLCD_RealWorld/REAL_TIME_WORKING/Models/ovrft/overfit8_900.pth'
    model = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    client_sock = connect_to_server()
    try:
        while True:
            inferr(device)
    except KeyboardInterrupt:
        client_sock.close()
