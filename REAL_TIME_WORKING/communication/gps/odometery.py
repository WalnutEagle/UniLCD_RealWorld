import math

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in kilometers. Use 3956 for miles
    r = 6371.0
    return c * r  # Distance in kilometers

def calculate_odometry(gps_data):
    previous_lat = None
    previous_lon = None
    total_distance = 0.0

    for data in gps_data:
        # Extract latitude and longitude from NMEA sentence
        # Assuming data is structured and includes the required information
        # For demonstration, we'll use hardcoded values
        lat = data['latitude']
        lon = data['longitude']

        if previous_lat is not None and previous_lon is not None:
            # Calculate distance from previous point
            distance = haversine(previous_lat, previous_lon, lat, lon)
            total_distance += distance
            print(f"Moved {distance:.4f} km. Total distance: {total_distance:.4f} km")

        # Update previous coordinates
        previous_lat = lat
        previous_lon = lon

    return total_distance

# Example GPS data (replace with real data parsing logic)
gps_data = [
    {'latitude': 37.7749, 'longitude': -122.4194},  # Point 1
    {'latitude': 37.7750, 'longitude': -122.4195},  # Point 2
    {'latitude': 37.7751, 'longitude': -122.4196},  # Point 3
]

if __name__ == "__main__":
    total_distance = calculate_odometry(gps_data)
    print(f"Total distance traveled: {total_distance:.4f} km")
