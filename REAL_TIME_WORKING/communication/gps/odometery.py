'''import math

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
print(f"Total distance traveled: {total_distance:.2f} meters")'''

import socket
import math

def parse_nmea_data(data):
    # Decode the byte data to string
    nmea_sentence = data.decode('utf-8').strip()

    latitude, longitude, speed = None, None, None

    if nmea_sentence.startswith('$GNRMC'):
        parts = nmea_sentence.split(',')
        if len(parts) > 5:
            latitude = float(parts[3]) * (1 if parts[4] == 'N' else -1)  # N/S indicator
            longitude = float(parts[5]) * (1 if parts[6] == 'E' else -1)  # E/W indicator
            speed = float(parts[7]) * 1.852  # Speed in knots to km/h (1 knot = 1.852 km/h)

    return latitude, longitude, speed

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

def start_udp_server():
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the address and port
    server_address = ('0.0.0.0', 8083)  # Listening on all interfaces
    sock.bind(server_address)

    print(f"Listening for UDP packets on {server_address[0]}:{server_address[1]}...")

    gps_data = []  # List to store GPS coordinates
    total_distance = 0.0  # Variable to track total distance traveled
    movement_threshold = 0.5  # Threshold in meters

    while True:
        # Wait for a message
        data, address = sock.recvfrom(4096)  # Buffer size is 4096 bytes
        # print(f"Received {data} from {address}")

        # Parse the NMEA data
        latitude, longitude, speed = parse_nmea_data(data)

        if latitude is not None and longitude is not None:
            gps_data.append((latitude, longitude))
            print(f"Latitude: {latitude}, Longitude: {longitude}, Speed: {speed:.2f} km/h")

            # Calculate distance since last point and total distance
            if len(gps_data) > 1:
                last_lat, last_lon = gps_data[-2]
                recent_distance = haversine(last_lat, last_lon, latitude, longitude)

                if recent_distance >= movement_threshold:  # Only consider significant changes
                    total_distance += recent_distance

                print(f"Distance traveled in last change: {recent_distance:.2f} meters")
                print(f"Total distance traveled: {total_distance:.2f} meters")

if __name__ == "__main__":
    start_udp_server()


