import math

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in meters (mean radius)
    r = 6371000
    return c * r

def calculate_total_distance(gps_data):
    total_distance = 0.0

    for i in range(1, len(gps_data)):
        lat1, lon1 = gps_data[i - 1]
        lat2, lon2 = gps_data[i]
        total_distance += haversine(lat1, lon1, lat2, lon2)

    return total_distance

# Example GPS data (latitude, longitude) every second
gps_data = [
    (37.7749, -122.4194),  # Point 1 (San Francisco)
    (37.7750, -122.4195),  # Point 2
    (37.7751, -122.4196),  # Point 3
    # Add more points as needed
]

total_distance = calculate_total_distance(gps_data)
print(f"Total distance traveled: {total_distance:.2f} meters")
