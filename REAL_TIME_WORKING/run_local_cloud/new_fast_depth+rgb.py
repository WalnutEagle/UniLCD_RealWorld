import os
import json
import datetime
import time
import argparse
import cv2
import depthai as dai
import numpy as np
import smbus2
import threading
from adafruit_servokit import ServoKit
import board
import busio
from pynput import keyboard as kb

i2c_bus0 = busio.I2C(board.SCL, board.SDA)
kit = ServoKit(channels=16, i2c=i2c_bus0, address=0x40)
kit.servo[0].set_pulse_width_range(1200, 2200)
kit.servo[1].set_pulse_width_range(1000, 2000)

# Global variables
throttle = 0.0
steer = 0.0
collect_data = False
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
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def map_value_throttle(x, in_min=-1, in_max=1, out_min=0, out_max=99):
    """Map values between (-1, 1) to (0, 99)"""
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def on_press(key):
    global throttle, steer, collect_data, exit_flag, create_new_directory
    if key == kb.Key.space:
        collect_data = not collect_data
    elif key == kb.KeyCode.from_char('n'):
        create_new_directory = True
    elif key == kb.KeyCode.from_char('w'):
        throttle = 1.0
    elif key == kb.KeyCode.from_char('s'):
        throttle = -1.0
    elif key == kb.KeyCode.from_char('a'):
        steer = -1.0
    elif key == kb.KeyCode.from_char('d'):
        steer = 1.0
    elif key == kb.KeyCode.from_char('q'):
        exit_flag = True
        return False

def on_release(key):
    global throttle, steer
    if key == kb.KeyCode.from_char('w') or key == kb.KeyCode.from_char('s'):
        throttle = 0.0
    elif key == kb.KeyCode.from_char('a') or key == kb.KeyCode.from_char('d'):
        steer = 0.0

def start_listener():
    with kb.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

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
    global throttle, steer, collect_data, create_new_directory, exit_flag
    bus_number = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="rc_data", help="Experiment name")
    args = parser.parse_args()
    frame_count = 0
    distance_to_obstacle = 0
    # Create directories for data storage
    dir_date = datetime.datetime.now().strftime("%m-%d-%Y")
    date_exp = os.path.join(dir_date, args.exp)
    if not os.path.exists(date_exp):
        os.makedirs(date_exp)
        full_path = os.path.join(date_exp, "run_001")
    else:
        directories = [d for d in os.listdir(date_exp) if os.path.isdir(os.path.join(date_exp, d))]
        run_number = len(directories) + 1
        full_path = os.path.join(date_exp, f"run_{run_number:03d}")
    os.makedirs(full_path, exist_ok=True)
    data_dir_rgb = os.path.join(full_path, "rgb")
    data_dir_disparity = os.path.join(full_path, "disparity")
    data_dir_json = os.path.join(full_path, "json")
    os.makedirs(data_dir_rgb, exist_ok=True)
    os.makedirs(data_dir_disparity, exist_ok=True)
    os.makedirs(data_dir_json, exist_ok=True)
    
    pipeline, depth = configure_depthai_pipeline()
    listener_thread = threading.Thread(target=start_listener)
    listener_thread.start()

    bus = smbus2.SMBus(bus_number)
    print(f"Connect to I2C Bus:{bus_number}")
    initialize_lidar(bus)

    power_mode = get_power_mode(bus)
    high_accuracy_mode = get_high_accuracy_mode(bus)

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_disparity = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
        start_measurement(bus)
        max_disparity = 255 
        while not exit_flag:
            start_time = time.time()
            
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                frame_rgb = in_rgb.getCvFrame()
                cv2.imshow("RGB", frame_rgb)

                if collect_data:
                    cv2.imwrite(f"{data_dir_rgb}/{frame_count:09d}_rgb.jpg", frame_rgb)

            in_disparity = q_disparity.tryGet()
            if in_disparity is not None:
                frame_disparity = in_disparity.getFrame()
                frame_disparity = (frame_disparity * (255 / max_disparity)).astype(np.uint8)
                cv2.imshow("Disparity", frame_disparity)

                if collect_data:
                    cv2.imwrite(f"{data_dir_disparity}/{frame_count:09d}_disparity.png", frame_disparity)
                    distance_to_obstacle = read_distance(bus)
                    sensor_data = {
                        'EpochTime': time.time(),
                        'DateTime': datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"),
                        'Throttle': throttle,
                        'Steer': steer,
                        'Distance': distance_to_obstacle,
                        }
                    json_str = json.dumps(sensor_data, indent=4)
                    with open(f"{data_dir_json}/{frame_count:09d}.json", 'w') as file:
                        file.write(json_str)
                frame_count += 1

            # if distance_to_obstacle<=55:
            #     mapped_steer = map_value_steer(0.0)
            #     mapped_throttle = map_value_throttle(0.0)
            # elif distance_to_obstacle>55:
            #     mapped_steer = map_value_steer(steer)
            #     mapped_throttle = map_value_throttle(throttle)
            mapped_steer = map_value_steer(steer)
            mapped_throttle = map_value_throttle(throttle)
            kit.servo[0].angle = mapped_steer
            kit.servo[1].angle = mapped_throttle

            if create_new_directory:
                print("Creating new directory.")
                directories = [d for d in os.listdir(date_exp) if os.path.isdir(os.path.join(date_exp, d))]
                run_number = len(directories) + 1
                full_path = os.path.join(date_exp, f"run_{run_number:03d}")
                os.makedirs(full_path, exist_ok=True)
                data_dir_rgb = os.path.join(full_path, "rgb")
                data_dir_disparity = os.path.join(full_path, "disparity")
                data_dir_json = os.path.join(full_path, "json")
                os.makedirs(data_dir_rgb, exist_ok=True)
                os.makedirs(data_dir_disparity, exist_ok=True)
                os.makedirs(data_dir_json, exist_ok=True)
                frame_count = 0
                create_new_directory = False

            # print(f"Total Time: {time.time() - start_time:.5f}")
            # print(create_new_directory)

    cv2.destroyAllWindows()
    listener_thread.join()

if __name__ == "__main__":
    main()