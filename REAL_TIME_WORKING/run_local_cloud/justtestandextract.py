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
from merger import get_preds
from missing import check_dataset, find_missing_files
from PIL import Image
from server import start_server, receive_data, send_response
model_path = "/home/h2x/Desktop/UniLCD_RealWorld/REAL_TIME_WORKING/run_local_cloud/model_run_0011.pth"
full_path = "/home/h2x/Desktop/REAL_TIME_WORKING/Main_script/10-11-2024/rc_data/run_001"
if __name__== "__main__":
    
    conn = start_server()
    t1 = time.time()
    output = get_preds(model_path, full_path)
    send_response(conn, output)
    serveroutput = receive_data(conn)
    print(f"Total Time taken is {time.time(0-t1)}seconds.")
    print(f"Yay the commms worked {serveroutput}")