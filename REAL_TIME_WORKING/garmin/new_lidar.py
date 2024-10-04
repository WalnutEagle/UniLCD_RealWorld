import smbus2
import time

LIDAR_ADDR = 0x62
REG_SYSRANGE_START = 0x00
REG_RESULT = 0x01
REG_RESULT_LOW_BYTE = 0x10
REG_RESULT_HIGH_BYTE = 0x11

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

def main():
    bus_number = 1
    try:
        bus = smbus2.SMBus(bus_number)
        print(f"Connected to I2C bus {bus_number}")
        initialize_lidar(bus)  
        
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

''' The below code all it does is avreages measurements and then gives the output'''
'''
def main():
    bus_number = 1
    num_measurements = 5
    try:
        bus = smbus2.SMBus(bus_number)
        print(f"Connected to I2C bus {bus_number}")
        initialize_lidar(bus)
        while True:
            start_measurement(bus)
            if wait_for_measurement(bus):
                distances = []
                for _ in range(num_measurements):
                    distance = read_distance(bus)
                    if distance is not None:
                        distances.append(distance)
                    time.sleep(0.1)
                if distances:
                    avg_distance = sum(distances) / len(distances)
                    print(f"Average Distance: {avg_distance:.2f} cm")
                else:
                    print("Failed to read distances.")
            else:
                print("Measurement failed.")
            time.sleep(1)
    except Exception as e:
        print(f"Error with I2C bus {bus_number}: {e}")
    finally:
        bus.close()
'''
if __name__ == "__main__":
    main()
