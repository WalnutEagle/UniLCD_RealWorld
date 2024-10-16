from jtop import jtop

def get_jetson_stats():
    with jtop() as jetson:
        cpu_usage = jetson.cpu
        gpu_usage = jetson.gpu
        power_usage = jetson.power

        print(f"CPU Usage: {cpu_usage}%")
        print(f"GPU Usage: {gpu_usage}%")
        if 'tot' in power_usage and 'power' in power_usage['tot']:
            total_power = power_usage['tot']['power'] / 1_000_000 
            print(f"Power Usage: {total_power:.2f} Watts")
        else:
            print("Power usage data not available.")

        print("-" * 40)  

if __name__ == "__main__":
    try:
        while True:
            get_jetson_stats()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
