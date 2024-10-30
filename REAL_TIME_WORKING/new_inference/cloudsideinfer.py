from cloudsidemodel import CustomRegNetY002
from nerc import connect_to_server, send_data, receive_response, client_loop
import torch
import numpy as np
import torch.nn as nn
import timm
import time
import glob
import os
import pickle
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

def inferr(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.to(device)
    with torch.no_grad():
        prediction = model(data)
    return prediction
    



if __name__ == '__main__': 
    model_path = '/opt/app-root/src/UniLCD_RealWorld/REAL_TIME_WORKING/Models/ovrft/overfit8_900.pth'
    model = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    socket_1 = connect_to_server()
    try:
        while True:
            
            response = receive_response(socket_1)
            if response is not None:
                tensord = inferr(response)
                print(tensord)
                send_data(socket_1, tensord)
            else :
                print('Trying again as didnt get anything')
    except KeyboardInterrupt:
        print('Bye')
        socket_1.close()

