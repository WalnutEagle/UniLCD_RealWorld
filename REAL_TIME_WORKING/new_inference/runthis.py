from power_stats.get_power import get_jetson_stats
if __name__ == '__main__':
    while True:
        well = get_jetson_stats()
        print(well)