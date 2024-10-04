1
import board
import busio
import time
from adafruit_servokit import ServoKit
import Jetson.GPIO as GPIO			       # Part of PWM DC motor control 
i2c_bus0=(busio.I2C(board.SCL, board.SDA))

kit = ServoKit(channels=16, i2c=i2c_bus0,address=0x40)

kit.servo[1].set_pulse_width_range(1000,2000)

while True:
	print(kit.servo[1].angle)
	user_input = input("Enter angle: ")
	kit.servo[1].angle = float(user_input)
	


'''import board
import busio
import time
from adafruit_servokit import ServoKit
import Jetson.GPIO as GPIO

i2c_bus0 = busio.I2C(board.SCL, board.SDA)
kit = ServoKit(channels=16, i2c=i2c_bus0, address=0x40)

kit.servo[1].set_pulse_width_range(1000, 2000)

while True:
    print("Current Angle:", kit.servo[1].angle)
    user_input = input("Enter angle (0-180): ")
    try:
        angle = int(user_input)
        if 0 <= angle <= 180:
            kit.servo[1].angle = angle
            print(f"Moving servo to angle: {angle}")
            time.sleep(1) 
        else:
            print("Angle must be between 0 and 180.")
    except ValueError:
        print("Please enter a valid integer.")'''
