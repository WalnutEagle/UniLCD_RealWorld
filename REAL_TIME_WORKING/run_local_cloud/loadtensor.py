import logging
import torch
import time
import numpy as np
tensor_path = '/opt/app-root/src/UniLCD_RealWorld/output_tensor.pt'  # Update with your actual path
t1 = time.time()
loaded_tensor = torch.load(tensor_path, map_location='cpu')
print(loaded_tensor)
print((time.time()-t1)*1000)