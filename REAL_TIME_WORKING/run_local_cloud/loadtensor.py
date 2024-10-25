import logging
import torch
import time
import numpy as np
from cloudsidemodel import CustomRegNetY00

def load_model(model_path):
    device = torch.device('cuda')
    checkpoint = torch.load(model_path, map_location=device)
    model = CustomRegNetY00()  # Initialize your model
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}  # Remove 'module.' prefix

    model.load_state_dict(state_dict)  # Load only the model state dict
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move model to the device
    return model

# Load the model once before the loop
model_path = "/opt/app-root/src/UniLCD_RealWorld/REAL_TIME_WORKING/run_local_cloud/model_run_0011.pth"
model = load_model(model_path)

device = torch.device('cuda')
tensor_path = '/opt/app-root/src/UniLCD_RealWorld/output_tensor.pt'  # Update with your actual path

while True:
    start = time.time()

    # Load the tensor directly to the CUDA device
    loaded_tensor = torch.load(tensor_path, map_location=device)

    with torch.no_grad():
        prediction = model(loaded_tensor)  # Predictions are being done on the GPU

    print(prediction)
    print(f"Total Time: {(time.time() - start) * 1000:.2f} milliseconds.")
