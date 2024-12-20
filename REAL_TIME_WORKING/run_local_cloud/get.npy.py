import torch
import numpy as np
from dataloader import CarlaRunDataset  # Ensure this is correct
import torch.nn as nn
import timm
import matplotlib.pyplot as plt
import time
import argparse
from torch.utils.data import DataLoader
from cloud1_model import CustomRegNetY002  # Adjust this to your actual model name
from dataloader import get_run_dataloader
import glob
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import json
import datetime
import cv2
from merger import get_preds, print_predictions
from missing import check_dataset, find_missing_files
from PIL import Image
from server import start_server, receive_data, send_response
# model_path = "/home/h2x/Desktop/UniLCD_RealWorld/REAL_TIME_WORKING/Models/ovrft/overfit8_900.pth"
model_path = "/home/h2x/Desktop/UniLCD_RealWorld/REAL_TIME_WORKING/run_local_cloud/model_run_0011.pth"
full_path = "/home/h2x/Desktop/UniLCD_RealWorld/REAL_TIME_WORKING/Main_script/10-17-2024/rc_data/run_001"
# full_path = "/home/h2x/Desktop/REAL_TIME_WORKING/Today's Data/allrun/10-03-2024/rc_data/run_001"
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
device = torch.device('cuda')
model = load_model(model_path)
test_dataset = CarlaRunDataset(full_path)  # Use your dataset class
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
while True:
    t1 = time.time()
    output = print_predictions(model, dataloader)
    print(time.time()-t1)
# print(output)
# output_tensor = torch.tensor(output)  # Ensure output is a NumPy array before this step
# torch.save(output_tensor, 'output_tensor.pt')  # Save the tensor to a file
# print("Tensor saved successfully.")
