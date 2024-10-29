import torch
import numpy as np
import torch.nn as nn
import timm
import matplotlib.pyplot as plt
import time
import argparse
import glob
import board
import busio
import os
import json
import datetime
import time
import argparse
import cv2
import depthai as dai
import numpy as np
import smbus2

from PIL import Image
from collections import deque
from pynput import keyboard as kb
from adafruit_servokit import ServoKit
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset

from merger import load_model
from missing import check_dataset, find_missing_files


i2c_bus0 = busio.I2C(board.SCL, board.SDA)
kit = ServoKit(channels=16, i2c=i2c_bus0, address=0x40)
kit.servo[0].set_pulse_width_range(1200, 2200)
kit.servo[1].set_pulse_width_range(1000, 2000)

# Global variables
throttle = 0.0
steer = 0.0
do_infer = False
create_new_directory = False
exit_flag = False  

LIDAR_ADDR = 0x62
REG_SYSRANGE_START = 0x00
REG_RESULT = 0x01
REG_RESULT_LOW_BYTE = 0x10
REG_RESULT_HIGH_BYTE = 0x11
REG_HIGH_ACCURACY_MODE = 0xEB
REG_POWER_MODE = 0xE2

def initialize_lidar(bus):
    pass

def start_measurement(bus):
    try:
        bus.write_byte_data(LIDAR_ADDR, REG_SYSRANGE_START, 0x04)
    except Exception as e:
        print(f"Error starting measurement: {e}")

def wait_for_measurement(bus):
    while True:
        try:
            status = bus.read_byte_data(LIDAR_ADDR, REG_RESULT)
            if status is not None:
                if (status & 0x01) == 0:
                    return True
        except Exception as e:
            print(f"Error reading status: {e}")
        time.sleep(0.01)

def read_distance(bus):
    try:
        low_byte = bus.read_byte_data(LIDAR_ADDR, REG_RESULT_LOW_BYTE)
        high_byte = bus.read_byte_data(LIDAR_ADDR, REG_RESULT_HIGH_BYTE)

        distance = (high_byte << 8) | low_byte
        return distance
    except Exception as e:
        print(f"Error reading distance: {e}")
        return None

def set_power_mode(bus, mode):
    try:
        bus.write_byte_data(LIDAR_ADDR, REG_POWER_MODE, mode)
        print(f"Power mode set to: {mode:#04x}")
    except Exception as e:
        print(f"Error setting power mode: {e}")

def set_high_accuracy_mode(bus, value):
    try:
        bus.write_byte_data(LIDAR_ADDR, REG_HIGH_ACCURACY_MODE, value)
        print(f"High accuracy mode set to: {value:#04x}")
    except Exception as e:
        print(f"Error setting high accuracy mode: {e}")

def get_power_mode(bus):
    try:
        power_mode = bus.read_byte_data(LIDAR_ADDR, REG_POWER_MODE)
        return power_mode
    except Exception as e:
        print(f"Error reading power mode: {e}")
        return None

def get_high_accuracy_mode(bus):
    try:
        high_accuracy_mode = bus.read_byte_data(LIDAR_ADDR, REG_HIGH_ACCURACY_MODE)
        return high_accuracy_mode
    except Exception as e:
        print(f"Error reading high accuracy mode: {e}")
        return None

def map_value_steer(x, in_min=-1, in_max=1, out_min=0, out_max=180):
    """Map values between (-1, 1) to (0, 180)"""
    return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def map_value_throttle(x, in_min=-1, in_max=1, out_min=0, out_max=99):
    """Map values between (-1, 1) to (0, 99)"""
    return float(x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def check_keys():
    global do_infer, exit_flag
    key = cv2.waitKey(1)  # Wait for 1 ms
    if key == ord(' '):  # Spacebar
        do_infer = not do_infer
    elif key == ord('q'):  # 'Q' key
        exit_flag = True


def configure_depthai_pipeline():
    pipeline = dai.Pipeline()
    
    # RGB Camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    # Stereo Depth
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)
    xout_depth = pipeline.create(dai.node.XLinkOut)

    xout_depth.setStreamName("disparity")

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")

    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    depth.setLeftRightCheck(True)
    depth.setExtendedDisparity(False)
    depth.setSubpixel(False)

    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.disparity.link(xout_depth.input)

    return pipeline, depth

def main():
    global throttle, steer, do_infer, create_new_directory, exit_flag
    bus_number = 1
    frame_count = 0
    distance_to_obstacle = 0

    pipeline, depth = configure_depthai_pipeline()

    bus = smbus2.SMBus(bus_number)
    print(f"Connect to I2C Bus:{bus_number}")
    initialize_lidar(bus)

    power_mode = get_power_mode(bus)
    high_accuracy_mode = get_high_accuracy_mode(bus)

    model_path = "/home/h2x/Desktop/REAL_TIME_WORKING/Overftmodels/Depth/overfit8_900.pth"

    model = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_disparity = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

        max_disparity = 255 
        while not exit_flag:
            check_keys()
            start_time = time.time()
            start_measurement(bus)
            if wait_for_measurement(bus):
                distance_to_obstacle = read_distance(bus)
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                frame_rgb = in_rgb.getCvFrame()
                cv2.imshow("RGB", frame_rgb)

            in_disparity = q_disparity.tryGet()
            if in_disparity is not None:
                frame_disparity = in_disparity.getFrame()
                frame_disparity = (frame_disparity * (255 / max_disparity)).astype(np.uint8)
                depth_img = torch.tensor(frame_disparity).float() / 255.0  # Normalize to [0, 1]
                depth_img = transforms.Resize((300, 300))(depth_img.unsqueeze(0))  # Resize
                depth_img = depth_img[0, :, :].unsqueeze(0)
                depth_img = depth_img.unsqueeze(0)
                depth_img = depth_img.to('cuda')
                sensor_data = {
                    'Throttle': throttle,
                    'Steer': steer,
                    }
                actions = torch.Tensor([sensor_data['Steer'],sensor_data['Throttle']])
                cv2.imshow("Disparity", frame_disparity)

                if do_infer:

                    s = time.time()
                    with torch.no_grad():
                        prediction = model(depth_img)
                    steering = prediction[0, 0].item()
                    throttle = prediction[0, 1].item()

                    print(f"Total Time: {time.time() - s:.5f}")
                    if distance_to_obstacle<=100:
                        mapped_steer = map_value_steer(0.0)
                        mapped_throttle = map_value_throttle(0.0)
                    else :
                        mapped_steer = map_value_steer(steer)
                        mapped_throttle = map_value_throttle(throttle)
                    if mapped_throttle > 99.0:
                        mapped_throttle = 99.0
                    elif mapped_throttle <0.0:
                        mapped_throttle = 0.0
                    print(f"steer {mapped_steer}, throttle {mapped_throttle}")
                    kit.servo[0].angle = mapped_steer
                    kit.servo[1].angle = mapped_throttle

                frame_count += 1
                if exit_flag:
                    throttle = 0.0
                    steer = 0.0
                    mapped_steer = map_value_steer(steer)
                    mapped_throttle = map_value_throttle(throttle)
                    kit.servo[1].angle = mapped_throttle
                    kit.servo[0].angle = mapped_steer

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
