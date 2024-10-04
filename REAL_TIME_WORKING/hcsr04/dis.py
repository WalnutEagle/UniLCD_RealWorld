'''import Jetson.GPIO as GPIO
import time

TRIG_PIN = 13
ECHO_PIN = 15

GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

def get_distance():
    GPIO.output(TRIG_PIN, GPIO.LOW)
    time.sleep(0.000002)

    GPIO.output(TRIG_PIN, GPIO.HIGH)
    time.sleep(0.00001) 
    GPIO.output(TRIG_PIN, GPIO.LOW)

    timeout = time.time() + 0.04
    while GPIO.input(ECHO_PIN) == GPIO.LOW and time.time() < timeout:
        pulse_start = time.time()
    
    if time.time() >= timeout:
        return -1 

    timeout = time.time() + 0.04 
    while GPIO.input(ECHO_PIN) == GPIO.HIGH and time.time() < timeout:
        pulse_end = time.time()
    
    if time.time() >= timeout:
        return -1 

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 34300 / 2
    return distance

try:
    while True:
        distance = get_distance()
        print(f"Distance: {distance:.2f} cm")
        time.sleep(1) 

except KeyboardInterrupt:
    print("Measurement stopped by user")
finally:
    GPIO.cleanup()'''



###################################################
##### AVERAGING JUST TO GET GOOD RESULTS###########
###################################################

'''import Jetson.GPIO as GPIO
import time

TRIG_PIN = 13
ECHO_PIN = 15

GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

def measure_single_distance():
    GPIO.output(TRIG_PIN, GPIO.LOW)
    time.sleep(0.000002)
    
    GPIO.output(TRIG_PIN, GPIO.HIGH)
    time.sleep(0.00001) 
    GPIO.output(TRIG_PIN, GPIO.LOW)

    timeout = time.time() + 0.04
    while GPIO.input(ECHO_PIN) == GPIO.LOW and time.time() < timeout:
        pulse_start = time.time()
    if time.time() >= timeout:
        return -1 
    timeout = time.time() + 0.04 
    while GPIO.input(ECHO_PIN) == GPIO.HIGH and time.time() < timeout:
        pulse_end = time.time()
    if time.time() >= timeout:
        return -1  
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 34300 / 2
    return distance

def get_distance(num_samples=5):
    distances = []
    for _ in range(num_samples):
        distance = measure_single_distance()
        if distance > 0:
            distances.append(distance)
        time.sleep(0.05)
    
    if distances:
        return sum(distances) / len(distances)
    return -1 

try:
    while True:
        distance = get_distance()
        if distance < 0:
            print("Error: Timeout occurred")
        else:
            print(f"Distance: {distance:.2f} cm")
        time.sleep(1)

except KeyboardInterrupt:
    print("Measurement stopped by user")

finally:
    GPIO.cleanup()
'''
################################################
####### FASTER OUTPUT FOR THE SAME CODE ########
################################################


import Jetson.GPIO as GPIO
import time

TRIG_PIN = 13
ECHO_PIN = 15

GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

def measure_single_distance():
    GPIO.output(TRIG_PIN, GPIO.LOW)
    time.sleep(0.000002) 
    GPIO.output(TRIG_PIN, GPIO.HIGH)
    time.sleep(0.00001) 
    GPIO.output(TRIG_PIN, GPIO.LOW)

    timeout = time.time() + 0.03
    while GPIO.input(ECHO_PIN) == GPIO.LOW and time.time() < timeout:
        pulse_start = time.time()
    
    if time.time() >= timeout:
        return -1

    timeout = time.time() + 0.03
    while GPIO.input(ECHO_PIN) == GPIO.HIGH and time.time() < timeout:
        pulse_end = time.time()
    
    if time.time() >= timeout:
        return -1

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 34300 / 2
    return distance

def get_distance(num_samples=5):
    distances = []
    for _ in range(num_samples):
        distance = measure_single_distance()
        if distance > 0:
            distances.append(distance)
        time.sleep(0.02) 
    if distances:
        return sum(distances) / len(distances)
    return -1 

try:
    while True:
        distance = get_distance()
        if distance < 0:
            print("Error: Timeout occurred")
        else:
            print(f"Distance: {distance:.1f} cm")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Measurement stopped by user")

finally:
    GPIO.cleanup()
