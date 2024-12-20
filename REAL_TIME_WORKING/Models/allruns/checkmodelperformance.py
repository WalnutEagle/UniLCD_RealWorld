import os
import glob
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from cloud1_dataloader import CarlaDataset  # Ensure this is correct
from cloud1_model import CustomRegNetY002  # Ensure this matches your model definition

def load_model(model_path):
    checkpoint = torch.load(model_path)  # Load the entire checkpoint
    model = CustomRegNetY002()  # Initialize your model
    
    # Strip 'module.' from the keys if the model was saved with Data Parallelism
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}  # Remove 'module.' prefix

    model.load_state_dict(state_dict)  # Load only the model state dict
    model.eval()  # Set the model to evaluation mode
    return model

'''The below part is just for small number of images'''

'''def visualize_predictions(model, dataloader, num_images=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_actuals = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient computation
        for i, (images, actions) in enumerate(dataloader):
            if i >= num_images:  # Limit the number of images to visualize
                break

            images = images.to(device)
            predictions = model(images).cpu().numpy()  # Forward pass to get predictions
            actions = actions.cpu().numpy()  # Get actual actions

            # Store actual and predicted actions
            all_actuals.extend(actions)
            all_predictions.extend(predictions)

    # Convert to numpy arrays for easier handling
    all_actuals = np.array(all_actuals)
    all_predictions = np.array(all_predictions)

    # Plot the predicted vs actual values
    plt.figure(figsize=(12, 6))
    
    # Plot steering
    plt.subplot(2, 1, 1)
    plt.plot(all_actuals[:, 0], label='Actual Steering', marker='o', markersize=4)
    plt.plot(all_predictions[:, 0], label='Predicted Steering', marker='x', markersize=4)
    plt.title('Steering: Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Steering Value')
    plt.legend()

    # Plot throttle
    plt.subplot(2, 1, 2)
    plt.plot(all_actuals[:, 1], label='Actual Throttle', marker='o', markersize=4)
    plt.plot(all_predictions[:, 1], label='Predicted Throttle', marker='x', markersize=4)
    plt.title('Throttle: Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Throttle Value')
    plt.legend()

    plt.tight_layout()
    plt.show()'''

def visualize_predictions(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_actuals = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient computation
        for images, actions in dataloader:
            images = images.to(device)
            predictions = model(images).cpu().numpy()  # Forward pass to get predictions
            actions = actions.cpu().numpy()  # Get actual actions

            # Store actual and predicted actions
            all_actuals.extend(actions)
            all_predictions.extend(predictions)

    # Convert to numpy arrays for easier handling
    all_actuals = np.array(all_actuals)
    all_predictions = np.array(all_predictions)

    # Plot the predicted vs actual values
    plt.figure(figsize=(12, 6))
    
    # Plot steering
    plt.subplot(2, 1, 1)
    plt.plot(all_actuals[:, 0], label='Actual Steering', marker='o', markersize=4)
    plt.plot(all_predictions[:, 0], label='Predicted Steering', marker='x', markersize=4)
    plt.title('Steering: Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Steering Value')
    plt.legend()

    # Plot throttle
    plt.subplot(2, 1, 2)
    plt.plot(all_actuals[:, 1], label='Actual Throttle', marker='o', markersize=4)
    plt.plot(all_predictions[:, 1], label='Predicted Throttle', marker='x', markersize=4)
    plt.title('Throttle: Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Throttle Value')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main(model_path, run_dir, batch_size=16):
    # Load the model
    model = load_model(model_path)

    # Create DataLoader for the new dataset
    test_dataset = CarlaDataset(run_dir)  # Use your dataset class
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Visualize predictions
    visualize_predictions(model, dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Imitation Learning DataSet Checking Script')
    parser.add_argument('-d', '--data_folder', default="/home/h2x/Desktop/REAL_TIME_WORKING/Today's Data/multiple people/10-03-2024", type=str, help='Path to your dataset')
    parser.add_argument('-m', '--model_path', default="/home/h2x/Desktop/REAL_TIME_WORKING/TrainedModels/following.pth", type=str, help='Path to your Trained Model')
    args = parser.parse_args()
    main(args.model_path, args.data_folder)


###############
## This is R2 Score part but doesnt work :(
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# def load_model(model_path):
#     checkpoint = torch.load(model_path)  # Load the entire checkpoint
#     model = CustomRegNetY002()  # Initialize your model
    
#     # Strip 'module.' from the keys if the model was saved with Data Parallelism
#     state_dict = checkpoint['model_state_dict']
#     if list(state_dict.keys())[0].startswith('module.'):
#         state_dict = {k[7:]: v for k, v in state_dict.items()}  # Remove 'module.' prefix

#     model.load_state_dict(state_dict)  # Load only the model state dict
#     model.eval()  # Set the model to evaluation mode
#     return model

# def evaluate_model(actuals, predictions):
#     mae = mean_absolute_error(actuals, predictions)
#     mse = mean_squared_error(actuals, predictions)
#     r2 = r2_score(actuals, predictions)

#     print(f"Mean Absolute Error (MAE): {mae:.4f}")
#     print(f"Mean Squared Error (MSE): {mse:.4f}")
#     print(f"R-squared (R²): {r2:.4f}")

# def visualize_predictions(model, dataloader, num_images=10):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)

#     all_actuals = []
#     all_predictions = []

#     with torch.no_grad():  # Disable gradient computation
#         for i, (images, actions) in enumerate(dataloader):
#             if i >= num_images:  # Limit the number of images to visualize
#                 break

#             images = images.to(device)
#             predictions = model(images).cpu().numpy()  # Forward pass to get predictions
#             actions = actions.cpu().numpy()  # Get actual actions

#             # Store actual and predicted actions
#             all_actuals.extend(actions)
#             all_predictions.extend(predictions)

#     # Convert to numpy arrays for easier handling
#     all_actuals = np.array(all_actuals)
#     all_predictions = np.array(all_predictions)

#     # Evaluate model performance
#     evaluate_model(all_actuals, all_predictions)

#     # Plot the predicted vs actual values
#     plt.figure(figsize=(12, 6))
    
#     # Plot steering
#     plt.subplot(2, 1, 1)
#     plt.plot(all_actuals[:, 0], label='Actual Steering', marker='o', markersize=4)
#     plt.plot(all_predictions[:, 0], label='Predicted Steering', marker='x', markersize=4)
#     plt.title('Steering: Actual vs Predicted')
#     plt.xlabel('Sample Index')
#     plt.ylabel('Steering Value')
#     plt.legend()

#     # Plot throttle
#     plt.subplot(2, 1, 2)
#     plt.plot(all_actuals[:, 1], label='Actual Throttle', marker='o', markersize=4)
#     plt.plot(all_predictions[:, 1], label='Predicted Throttle', marker='x', markersize=4)
#     plt.title('Throttle: Actual vs Predicted')
#     plt.xlabel('Sample Index')
#     plt.ylabel('Throttle Value')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# def main(model_path, run_dir, batch_size=16):
#     # Load the model
#     model = load_model(model_path)

#     # Create DataLoader for the new dataset
#     test_dataset = CarlaDataset(run_dir)  # Use your dataset class
#     dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     # Visualize predictions
#     visualize_predictions(model, dataloader)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Imitation Learning DataSet Checking Script')
#     parser.add_argument('-d', '--data_folder', default="/home/h2x/Desktop/REAL_TIME_WORKING/Today's Data/multiple people/10-03-2024", type=str, help='Path to your dataset')
#     parser.add_argument('-m', '--model_path', default="/home/h2x/Desktop/REAL_TIME_WORKING/TrainedModels/following.pth", type=str, help='Path to your Trained Model')
#     args = parser.parse_args()
#     main(args.model_path, args.data_folder)