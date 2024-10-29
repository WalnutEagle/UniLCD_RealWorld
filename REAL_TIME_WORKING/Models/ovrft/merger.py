import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from cloud1_dataloader import CarlaDataset  # Ensure this is correct
from cloud1_model import CustomRegNetY002  # Ensure this matches your model definition

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

def print_predictions(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_predictions = []

    with torch.no_grad():  # Disable gradient computation
        for images, actions in dataloader:
            images = images.to(device)
            predictions = model(images).cpu().numpy()  # Forward pass to get predictions
            
            # Store predictions
            all_predictions.extend(predictions)

    # Convert to numpy array for easier handling
    all_predictions = np.array(all_predictions)
    return all_predictions
    # Print predictions
    # for i, prediction in enumerate(all_predictions):
    #     print(f"Sample {i}: Predicted Steering: {prediction[0]}, Predicted Throttle: {prediction[1]}")

def get_preds(model_path, run_dir, batch_size=16):
    # Load the model
    model = load_model(model_path)

    # Create DataLoader for the new dataset
    test_dataset = CarlaDataset(run_dir)  # Use your dataset class
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print predictions
    return print_predictions(model, dataloader)

if __name__ == "__main__":
    model_path = "/home/h2x/Desktop/trainedmodels/model_run_0011.pth"  # Update with the path to your trained model
    run_dir = "/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024/rc_data/run_006"
    get_preds(model_path, run_dir)
