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
import cv2
import depthai as dai
import smbus2
from PIL import Image
from collections import deque
from pynput import keyboard as kb
from adafruit_servokit import ServoKit
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib.patches import Arc, Circle

from merger import load_model
from missing import check_dataset, find_missing_files

# Initialize I2C and ServoKit
i2c_bus0 = busio.I2C(board.SCL, board.SDA)
kit = ServoKit(channels=16, i2c=i2c_bus0, address=0x40)
kit.servo[0].set_pulse_width_range(1200, 2200)
kit.servo[1].set_pulse_width_range(1000, 2000)

# Global variables
throttle = 0.0
steer = 0.0
do_infer = False
exit_flag = False
current_steer = 0.0  # Define default value
current_throttle = 0.0  # Define default value
current_mode = "Local Mode"

def create_speedometer(ax, current_value, title):
    ax.clear()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    arc_background = Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=180, color='lightgrey', lw=10)
    ax.add_patch(arc_background)

    theta2 = current_value * 180 / 100
    arc_fill = Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=theta2, color='blue', lw=10)
    ax.add_patch(arc_fill)

    for i in range(0, 101, 10):
        angle = np.radians(i * 180 / 100)
        x = np.cos(angle)
        y = np.sin(angle)
        ax.text(x * 1.1, y * 1.1, str(i), ha='center', va='center', fontsize=8)

    ax.text(0, -0.2, title, ha='center', va='center', fontsize=12)

def draw_mode_indicator(ax):
    mode_color = 'green' if current_mode == 'Local Mode' else 'yellow'
    ax.clear()
    ax.add_patch(Circle((0.5, 0.5), 2, color=mode_color))
    ax.text(0.5, 0.2, current_mode, fontsize=12, va='center', ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def update_mode(mode):
    global current_mode
    current_mode = 'Cloud Mode' if mode == 1 else 'Local Mode'

def update_visualization():
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    mode_ax = fig.add_axes([0.4, 0.05, 0.2, 0.1])

    plt.ion()

    while not exit_flag:
        global current_throttle, current_steer
        create_speedometer(ax1, current_steer, 'Steer')
        create_speedometer(ax2, current_throttle, 'Throttle')
        draw_mode_indicator(mode_ax)
        
        plt.pause(0.05)

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

def map_value_steer(x, in_min=-1, in_max=1, out_min=0, out_max=180):
    return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def map_value_throttle(x, in_min=-1, in_max=1, out_min=0, out_max=99):
    return float(x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def check_keys():
    global do_infer, exit_flag
    key = cv2.waitKey(1)
    if key == ord(' '):
        do_infer = not do_infer
    elif key == ord('q'):
        exit_flag = True

def configure_depthai_pipeline():
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

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
    global throttle, steer, do_infer, exit_flag, current_steer, current_throttle

    bus_number = 1
    frame_count = 0
    distance_to_obstacle = 0

    pipeline, depth = configure_depthai_pipeline()
    bus = smbus2.SMBus(bus_number)
    print(f"Connect to I2C Bus:{bus_number}")
    initialize_lidar(bus)

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
                frame_disparity = cv2.applyColorMap(frame_disparity, cv2.COLORMAP_JET)
                cv2.imshow("Disparity", frame_disparity)

            if do_infer:
                inference_start_time = time.time()
                input_image = transforms.ToTensor()(frame_rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_image)
                throttle, steer = output.cpu().numpy()[0]
                current_throttle = map_value_throttle(throttle)
                current_steer = map_value_steer(steer)
                update_visualization()

            frame_count += 1
            time.sleep(0.01)

if __name__ == "__main__":
    main()
