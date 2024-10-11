from depthai import camera
from mode import calc
import cv2
from missing import check_dataset, find_missing_files
i2c_bus0 = busio.I2C(board.SCL, board.SDA)
kit = ServoKit(channels=16, i2c=i2c_bus0, address=0x40)
kit.servo[0].set_pulse_width_range(1200, 2200)
kit.servo[1].set_pulse_width_range(1000, 2000)

if __name__ == "__camera__":
    camera()
    calc()
    for frame in camera():
        # Process the frame
        cv2.imshow("RGB Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    for distnace in calc():
        print(f"Distance is {distnace}")
    