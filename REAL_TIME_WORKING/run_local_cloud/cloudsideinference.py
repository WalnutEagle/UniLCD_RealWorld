import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from cloudsidemodel import CustomRegNetY00  # Ensure this matches your model definition
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    all_predictions = []
    with torch.no_grad():
        predictions = model(images).cpu().numpy()  # Forward pass to get predictions 
        all_predictions.extend(predictions)
    all_predictions = np.array(all_predictions)
    return all_predictions
def get_preds(model_path,predictions):
    model = load_model(model_path)
    return inference(model, predictions)

if __name__ == "__main__":
    model_path = "/home/h2x/Desktop/trainedmodels/model_run_0011.pth"  # Update with the path to your trained model
    run_dir = "/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024/rc_data/run_006"
    get_preds(model_path, run_dir)

