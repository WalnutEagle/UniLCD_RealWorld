import socket
import json
import math

# Server configuration
server_ip = '10.239.28.208'
server_port = 8080

# Haversine formula for distance in meters
def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Distance in meters

# Initialize variables
prev_lat = None
prev_lon = None
total_distance = 0.0  # in meters
current_x, current_y = 0.0, 0.0  # start at (0,0)

# Connect to the server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))
print(f"Connected to server {server_ip} on port {server_port}")

try:
    while True:
        data = client_socket.recv(4096)
        if not data:
            break  # If no data, exit loop

        # Try to decode JSON data if received
        try:
            gps_data = json.loads(data.decode('utf-8'))
        except json.JSONDecodeError:
            continue  # Ignore non-JSON messages

        # Extract GPS coordinates, speed, and heading
        latitude = gps_data['latitude']
        longitude = gps_data['longitude']
        heading = gps_data['heading']  # in degrees
        speed_kmh = gps_data['speed']  # already in km/h

        # Calculate distance traveled if there was a previous location
        if prev_lat is not None and prev_lon is not None:
            distance = haversine(prev_lat, prev_lon, latitude, longitude)
            total_distance += distance  # Update total distance

            # Convert distance and heading to x and y coordinates
            rad_heading = math.radians(heading)
            delta_x = distance * math.cos(rad_heading)
            delta_y = distance * math.sin(rad_heading)

            # Update current position
            current_x += delta_x
            current_y += delta_y

            # Print updates
            print(f"New position (lat, lon): ({latitude}, {longitude})")
            print(f"Distance travelled in last update: {distance:.2f} meters")
            print(f"Total distance travelled: {total_distance:.2f} meters")
            print(f"Current x, y position: ({current_x:.2f}, {current_y:.2f})")
            print(f"Speed: {speed_kmh:.2f} km/h")
            print(f"Heading: {heading:.2f} degrees\n")
        
        # Update previous latitude and longitude
        prev_lat, prev_lon = latitude, longitude

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    client_socket.close()
    print("Connection closed.")
