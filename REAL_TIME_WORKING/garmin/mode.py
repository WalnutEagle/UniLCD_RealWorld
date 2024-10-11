import smbus2
import time

LIDAR_ADDR = 0x62
REG_SYSRANGE_START = 0x00
REG_RESULT = 0x01
REG_RESULT_LOW_BYTE = 0x10
REG_RESULT_HIGH_BYTE = 0x11
REG_HIGH_ACCURACY_MODE = 0xEB
REG_POWER_MODE = 0xE2 

def initialize_lidar(bus):
    set_power_mode(bus, 0xFF) 
    set_high_accuracy_mode(bus, 0x14)  

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

def distance():
    bus_number = 1
    try:
        bus = smbus2.SMBus(bus_number)
        print(f"Connected to I2C bus {bus_number}")
        initialize_lidar(bus)
        power_mode = get_power_mode(bus)
        high_accuracy_mode = get_high_accuracy_mode(bus)
        print(f"Power Mode: {power_mode}")  
        print(f"High Accuracy Mode: {high_accuracy_mode}")

        while True:
            start_measurement(bus)
            if wait_for_measurement(bus):
                distance = read_distance(bus)
                if distance is not None:
                    print(f"Distance: {distance} cm")
                else:
                    print("Failed to read distance.")
            else:
                print("Measurement failed.")
            
            time.sleep(1)  

    except Exception as e:
        print(f"Error with I2C bus {bus_number}: {e}")
    finally:
        bus.close()

if __name__ == "__main__":
    distance()
