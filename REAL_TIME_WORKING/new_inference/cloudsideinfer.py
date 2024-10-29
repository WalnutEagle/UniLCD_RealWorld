from cloudsidemodel import CustomRegNetY002
from client import connect_to_server, send_data , receive_response
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
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



model_path = "/home/h2x/Desktop/REAL_TIME_WORKING/Overftmodels/Depth/overfit8_900.pth"
model = load_model(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)
client_socket = connect_to_server()
send_data(client_socket, 'a')

def inferr():
    data = receive_response(client_socket)
    data.to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prediction = model(data)
    send_data(client_socket, prediction)

