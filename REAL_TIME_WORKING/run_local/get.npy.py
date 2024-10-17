import torch
import numpy as np
  # Ensure this is correct
import torch.nn as nn
import timm
import matplotlib.pyplot as plt
import time
import argparse
from torch.utils.data import DataLoader
from cloud1_model import CustomRegNetY002  # Adjust this to your actual model name
from dataloader import get_run_dataloader

from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset

from merger import get_preds
from missing import check_dataset, find_missing_files

model_path = "/home/h2x/Desktop/UniLCD_RealWorld/REAL_TIME_WORKING/Models/ovrft/overfit8_900.pth"
full_path = "/home/h2x/Desktop/REAL_TIME_WORKING/Main_script/10-17-2024/rc_data/run_001"
output = get_preds(model_path, full_path)
print(output)
np.save('well.npy', output)