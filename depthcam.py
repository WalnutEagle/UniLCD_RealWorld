import depthai
import cv2
import time

def main():
    # Create pipeline
    pipeline = depthai.Pipeline()

    # Define sources and outputs
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setBoardSocket(depthai.CameraBoardSocket.CAM_A)
    #cam_rgb.setFps(30)

    mono_left = pipeline.createMonoCamera()
    mono_left.setBoardSocket(depthai.CameraBoardSocket.CAM_B)
    #mono_left.setFps(30)
 

    mono_right = pipeline.createMonoCamera()
    mono_right.setBoardSocket(depthai.CameraBoardSocket.CAM_C)
    #mono_right.setFps(30)


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

    # Connect
    with depthai.Device(pipeline) as device:
        # Output
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        # device.setFps(30)

        while True:
            in_rgb = q_rgb.tryGet()
            in_depth = q_depth.tryGet()

            if in_rgb is not None:
                # Get RGB frame
                frame_rgb = in_rgb.getCvFrame()

                # Get depth frame
                if in_depth is not None:
                    frame_depth = in_depth.getFrame()
                    timestamp = int(time.time())
                    cv2.imwrite(f"test_data/rgb_{timestamp}.jpg", frame_rgb)
                    cv2.imwrite(f"test_data/depth_{timestamp}.png", frame_depth)

                cv2.imshow("RGB", frame_rgb)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()