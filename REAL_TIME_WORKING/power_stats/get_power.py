'''from jtop import jtop

def get_jetson_stats():
    with jtop() as jetson:
        cpu_usage = jetson.cpu
        gpu_usage = jetson.gpu
        power_usage = jetson.power

        print(f"CPU Usage: {cpu_usage}%")
        print(f"GPU Usage: {gpu_usage}%")
        print(f"Power Usage: {power_usage['power'] / 1_000_000:.2f} Watts")
        print("-" * 40)

if __name__ == "__main__":
    try:
        while True:
            get_jetson_stats()
    except KeyboardInterrupt:
        print("\nMonitoring Stopped by user:)")
'''

from jtop import jtop
import time

def get_jetson_stats():
    with jtop() as jetson:
        cpu_usage = jetson.cpu
        gpu_usage = jetson.gpu
        memory_usage = jetson.memory
        power_usage = jetson.power

        print(f"CPU Usage: {cpu_usage}%")
        print(f"GPU Usage: {gpu_usage}%")

        # Check memory usage structure
        if 'used' in memory_usage and 'total' in memory_usage:
            memory_used = memory_usage['used']
            memory_total = memory_usage['total']
            memory_percent = (memory_used / memory_total) * 100
            print(f"Memory Usage: {memory_percent:.2f}%")
        else:
            print("Memory usage data not available.")

        # Print entire power usage data for inspection
        print("Power Usage Data:", power_usage)  # Inspect the structure
        
        # Access power usage if the expected key exists
        if 'power' in power_usage:
            print(f"Power Usage: {power_usage['power'] / 1_000_000:.2f} Watts")  # Convert microWatts to Watts
        else:
            print("Power usage data not available.")
        
        print("-" * 40)  # Separator for readability

if __name__ == "__main__":
    try:
        while True:
            get_jetson_stats()
            time.sleep(1)  # Pause for a second between updates
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
