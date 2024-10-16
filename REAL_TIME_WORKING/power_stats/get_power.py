from jtop import jtop

def get_jetson_stats():
    with jtop() as jetson:

        power_usage = jetson.power
        if 'tot' in power_usage and 'power' in power_usage['tot']:
            total_power = power_usage['tot']['power'] / 1000 
            print(f"Power Usage: {total_power:.2f} Watts")
        else:
            print("Power usage data not available.")

if __name__ == "__main__":
    try:
        while True:
            get_jetson_stats()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
