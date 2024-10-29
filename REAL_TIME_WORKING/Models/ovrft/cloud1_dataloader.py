import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import os
import json
import time
import argparse
from Models.ovrft.cloud1_model import CustomRegNetY002
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the dataset class
class CarlaDataset(Dataset):
    def __init__(self, data_dir):
        self.img_list = []
        self.depth_list = []
        self.data_list = []
        self.data_dir = data_dir

        # Traverse through subdirectories
        for run_dir in glob.glob(os.path.join(self.data_dir, 'rc_data', 'run_*')):
            # Load RGB images
            rgb_dir = os.path.join(run_dir, 'rgb')
            self.img_list += glob.glob(os.path.join(rgb_dir, '*.jpg'))

            # Load Depth images
            depth_dir = os.path.join(run_dir, 'disparity')
            self.depth_list += glob.glob(os.path.join(depth_dir, '*.png'))

            # Load JSON files for actions
            json_dir = os.path.join(run_dir, 'json')
            self.data_list += glob.glob(os.path.join(json_dir, '*.json'))

        min_length = min(len(self.img_list), len(self.depth_list), len(self.data_list))
        self.img_list = self.img_list[:min_length]
        self.depth_list = self.depth_list[:min_length]
        self.data_list = self.data_list[:min_length]

        logging.info(f'Loaded {len(self.img_list)} images, {len(self.depth_list)} depth images, and {len(self.data_list)} action files.')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            with open(self.data_list[idx], 'r') as f:
                data = json.load(f)

            # Extract actions (steering and throttle)
            actions = torch.Tensor([data["Steer"], data["Throttle"]])

            # Load the corresponding depth image (ignore RGB)
            depth_path = self.depth_list[idx]
            depth_img = read_image(depth_path)
            depth_img = depth_img.float() / 255.0  # Normalize depth to [0, 1]

            # Resize the depth image (remove normalization/combination with RGB)
            depth_img = transforms.Resize((300, 300))(depth_img)

            # Optionally: Convert depth to grayscale by keeping one channel (if required by your model)
            depth_img = depth_img[0, :, :].unsqueeze(0)  # Keep only one channel if it's a grayscale depth map

            return depth_img, actions 
        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            raise


###### https://drive.google.com/drive/folders/1KwRo-KWG-NsiULJ3RpXggJrZfAHTL8J3?usp=sharing