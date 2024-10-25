import logging
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from cloudsidemodel import CustomRegNetY00  # Ensure this matches your model definition
def load_model(model_path):
    device = torch.device('cuda')
    print(device)
    checkpoint = torch.load(model_path, map_location=device)  # Load the entire checkpoint
    model = CustomRegNetY00()  # Initialize your model
    # state_dict = checkpoint
    # Strip 'module.' from the keys if the model was saved with Data Parallelism
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}  # Remove 'module.' prefix

    model.load_state_dict(state_dict)  # Load only the model state dict
    model.eval()  # Set the model to evaluation mode
    return model

def inference(model,predictions):
    device = torch.device('cuda')
    model.to(device)

    with torch.no_grad():
        prediction = model(predictions)  # Forward pass to get predictions 

    return prediction
def get_preds(model_path,predictions):
    model = load_model(model_path)
    return inference(model, predictions)

if __name__ == "__main__":
    tensor_path = '/opt/app-root/src/UniLCD_RealWorld/output_tensor.pt'  # Update with your actual path
    loaded_tensor = torch.load(tensor_path, map_location='cuda')
    model_path = "/opt/app-root/src/UniLCD_RealWorld/REAL_TIME_WORKING/run_local_cloud/model_run_0011.pth"  # Update with the path to your trained model
    while True:
        start = time.time()
        output = get_preds(model_path, loaded_tensor)
        print(f"Total Inference Time is:{(time.time()-start)*1000}Miliseconds")
        print(output)

