import depthai
import cv2
import time
import os
import json
import datetime
import argparse
import numpy as np


def main():
    throttle = 0.0
    steer = 0.0
    collect_data = False  # Initially not collecting data
    create_new_directory = False

    # Create pipeline
    pipeline = depthai.Pipeline()

    # Define sources and outputs
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setBoardSocket(depthai.CameraBoardSocket.CAM_A)

    mono_left = pipeline.createMonoCamera()
    mono_left.setBoardSocket(depthai.CameraBoardSocket.CAM_B)

    mono_right = pipeline.createMonoCamera()
    mono_right.setBoardSocket(depthai.CameraBoardSocket.CAM_C)

    stereo = pipeline.createStereoDepth()
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="rc_data", help="Experiment name")
    args = parser.parse_args()
    frame_count = 0

    max_disparity = 255
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

    # Connect to the device
    with depthai.Device(pipeline) as device:
        # Output
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        while True:
            # Check for user input
            key = cv2.waitKey(1) & 0xFF
            if key == 32: 
                collect_data = not collect_data 
            if key == ord('q'):  
                break
            if key == ord('n'):
                create_new_directory=True
                
            in_rgb = q_rgb.get()
            in_disparity = q_depth.tryGet()

            if in_rgb is not None:
                frame_rgb = in_rgb.getCvFrame()
                # cv2.imshow("RGB", frame_rgb)
                yield frame_rgb
 
                # Save RGB frame if data collection is acti
                if in_disparity is not None:
                    frame_disparity = in_disparity.getFrame()
                    frame_disparity = (frame_disparity * (255 / max_disparity)).astype(np.uint8)
                    if collect_data:
                        cv2.imwrite(f"{data_dir_rgb}/{frame_count:09d}_rgb.jpg", frame_rgb)
                        # Save disparity frame
                        cv2.imwrite(f"{data_dir_disparity}/{frame_count:09d}_disparity.png", frame_disparity)

                        # Create and save sensor data in JSON format
                        sensor_data = {
                            'EpochTime': time.time(),
                            'DateTime': datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"),
                            'Throttle': throttle,
                            'Steer': steer,
                            'Distance': 0.0  # Replace with actual distance sensor value if available
                        }
                        json_str = json.dumps(sensor_data, indent=4)
                        with open(f"{data_dir_json}/{frame_count:09d}.json", 'w') as file:
                            file.write(json_str)

                frame_count += 1
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

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
