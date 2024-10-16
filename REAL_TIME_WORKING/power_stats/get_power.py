from jtop import jtop

def get_jetson_stats():
    with jtop() as jetson:
        cpu_usage = jetson.cpu
        gpu_usage = jetson.gpu
        memory_usage = jetson.memory
        power_usage = jetson.power

        print(f"CPU Usage: {cpu_usage}%")
        print(f"GPU Usage: {gpu_usage}%")
        print(f"Memory Usage: {memory_usage['used'] / memory_usage['total'] * 100:.2f}%")
        print(f"Power Usage: {power_usage['power'] / 1_000_000:.2f} Watts") 

if __name__ == "__main__":
    get_jetson_stats()
