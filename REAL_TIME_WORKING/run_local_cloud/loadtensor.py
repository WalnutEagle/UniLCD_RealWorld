import logging
import torch
import time
import numpy as np
tensor_path = '/opt/app-root/src/UniLCD_RealWorld/output_tensor.pt'  # Update with your actual path
start = time.time()
loaded_tensor = torch.load(tensor_path, map_location='cuda')
model_path = "/opt/app-root/src/UniLCD_RealWorld/REAL_TIME_WORKING/run_local_cloud/model_run_0011.pth"  # Update with the path to your trained model
model = load_model(model_path)
with torch.no_grad():
    prediction = model(loaded_tensor)
print(f"Total Time{(time.time()-start)*1000}Miliseconds.")