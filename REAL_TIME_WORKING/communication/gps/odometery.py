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

# import socket
# import math

# def parse_nmea_data(data):
#     # Decode the byte data to string
#     nmea_sentence = data.decode('utf-8').strip()

#     latitude, longitude, speed = None, None, None

#     try:
#         if nmea_sentence.startswith('$GNRMC'):
#             parts = nmea_sentence.split(',')
#             if len(parts) > 5:
#                 latitude = float(parts[3]) * (1 if parts[4] == 'N' else -1)  # N/S indicator
#                 longitude = float(parts[5]) * (1 if parts[6] == 'E' else -1)  # E/W indicator
#                 speed = float(parts[7]) * 1.852  # Speed in knots to km/h (1 knot = 1.852 km/h)
#     except ValueError :
#         pass

#     return latitude, longitude, speed

# def haversine(lat1, lon1, lat2, lon2):
#     # Convert latitude and longitude from degrees to radians
#     lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

#     # Haversine formula
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
#     c = 2 * math.asin(math.sqrt(a))
    
#     # Radius of Earth in meters (mean radius)
#     r = 6371000
#     return c * r

# def calculate_total_distance(gps_data):
#     total_distance = 0.0

#     for i in range(1, len(gps_data)):
#         lat1, lon1 = gps_data[i - 1]
#         lat2, lon2 = gps_data[i]
#         total_distance += haversine(lat1, lon1, lat2, lon2)

#     return total_distance

# def start_udp_server():
#     # Create a UDP socket
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#     # Bind the socket to the address and port
#     server_address = ('0.0.0.0', 8083)  # Listening on all interfaces
#     sock.bind(server_address)

#     print(f"Listening for UDP packets on {server_address[0]}:{server_address[1]}...")

#     gps_data = []  # List to store GPS coordinates
#     total_distance = 0.0  # Variable to track total distance traveled
#     movement_threshold = 0.5  # Threshold in meters

#     while True:
#         # Wait for a message
#         data, address = sock.recvfrom(4096)  # Buffer size is 4096 bytes
#         # print(f"Received {data} from {address}")

#         # Parse the NMEA data
#         latitude, longitude, speed = parse_nmea_data(data)

#         if latitude is not None and longitude is not None:
#             gps_data.append((latitude, longitude))
#             print(f"Latitude: {latitude}, Longitude: {longitude}, Speed: {speed:.2f} km/h")

#             # Calculate distance since last point and total distance
#             if len(gps_data) > 1:
#                 last_lat, last_lon = gps_data[-2]
#                 recent_distance = haversine(last_lat, last_lon, latitude, longitude)

#                 if recent_distance >= movement_threshold:  # Only consider significant changes
#                     total_distance += recent_distance

#                 print(f"Distance traveled in last change: {recent_distance:.2f} meters")
#                 print(f"Total distance traveled: {total_distance:.2f} meters")

# if __name__ == "__main__":
#     start_udp_server()








'''import socket
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

# Define thresholds
SPEED_THRESHOLD = 0.0  # Minimum speed in km/h to consider a valid movement
DISTANCE_THRESHOLD = 1.0  # Minimum distance in meters to update position

# Create and connect the socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((server_ip, server_port))
    print(f"Connected to server {server_ip} on port {server_port}")

    while True:
        # Receive data
        data = client_socket.recv(4096)
        if not data:
            break  # Exit if there's no more data

        # Decode received data
        decoded_data = data.decode('utf-8').strip()
        
        # Parse NMEA sentences
        if decoded_data.startswith('$'):
            nmea_sentence = decoded_data.split(',')
            if nmea_sentence[0] == '$GPGGA' and len(nmea_sentence) >= 6:
                # Extract latitude and longitude from GPGGA
                lat = float(nmea_sentence[2]) / 100.0  # Convert to decimal
                lon = float(nmea_sentence[4]) / 100.0  # Convert to decimal

                # Convert latitude and longitude from DMS to decimal degrees
                lat_deg = int(lat / 100)
                lat_min = lat % 100
                latitude = lat_deg + lat_min / 60.0
                if nmea_sentence[3] == 'S':
                    latitude = -latitude  # South is negative

                lon_deg = int(lon / 100)
                lon_min = lon % 100
                longitude = lon_deg + lon_min / 60.0
                if nmea_sentence[5] == 'W':
                    longitude = -longitude  # West is negative

                # Update previous coordinates for distance calculation
                if prev_lat is not None and prev_lon is not None:
                    distance = haversine(prev_lat, prev_lon, latitude, longitude)
                    
                    # Only consider valid movements based on speed and distance thresholds
                    if last_speed >= SPEED_THRESHOLD and distance >= DISTANCE_THRESHOLD:
                        total_distance += distance

                        # Update current x and y position using last known heading
                        print(last_speed)
                        print('Working on as the last speed is less than the threshold')
                        rad_heading = math.radians(last_heading) if 'last_heading' in locals() else 0
                        delta_x = distance * math.cos(rad_heading)
                        delta_y = distance * math.sin(rad_heading)

                        current_x += delta_x
                        current_y += delta_y

                        # Display data
                        print(f"New position (lat, lon): ({latitude:.6f}, {longitude:.6f})")
                        print(f"Distance travelled in last update: {distance:.2f} meters")
                        print(f"Total distance travelled: {total_distance:.2f} meters")
                        print(f"Current x, y position: ({current_x:.2f}, {current_y:.2f})\n")

                    # Update previous coordinates
                    prev_lat, prev_lon = latitude, longitude

            elif nmea_sentence[0] == '$GPRMC' and len(nmea_sentence) >= 9:
                # Extract speed and heading from GPRMC
                last_speed = float(nmea_sentence[7]) * 1.852  # Convert knots to km/h
                last_heading = float(nmea_sentence[8])  # Heading in degrees

                # You can print speed and heading if needed
                print(f"Speed: {last_speed:.2f} km/h, Heading: {last_heading:.2f} degrees")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    client_socket.close()
    print("Connection closed.")
'''





import socket
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

# Define thresholds
SPEED_THRESHOLD = 2.0  # Minimum speed in km/h to consider a valid movement
DISTANCE_THRESHOLD = 1.0  # Minimum distance in meters to update position

# Create and connect the socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((server_ip, server_port))
    print(f"Connected to server {server_ip} on port {server_port}")

    while True:
        # Receive data
        data = client_socket.recv(4096)
        if not data:
            break  # Exit if there's no more data

        # Decode received data
        decoded_data = data.decode('utf-8').strip()
        
        # Parse NMEA sentences
        if decoded_data.startswith('$'):
            nmea_sentence = decoded_data.split(',')
            if nmea_sentence[0] == '$GPGGA' and len(nmea_sentence) >= 6:
                # Extract latitude and longitude from GPGGA
                lat = float(nmea_sentence[2]) / 100.0  # Convert to decimal
                lon = float(nmea_sentence[4]) / 100.0  # Convert to decimal

                # Convert latitude and longitude from DMS to decimal degrees
                lat_deg = int(lat / 100)
                lat_min = lat % 100
                latitude = lat_deg + lat_min / 60.0
                if nmea_sentence[3] == 'S':
                    latitude = -latitude  # South is negative

                lon_deg = int(lon / 100)
                lon_min = lon % 100
                longitude = lon_deg + lon_min / 60.0
                if nmea_sentence[5] == 'W':
                    longitude = -longitude  # West is negative

                # Update previous coordinates for distance calculation
                if prev_lat is not None and prev_lon is not None:
                    distance = haversine(prev_lat, prev_lon, latitude, longitude)

                    # Only consider valid movements based on speed and distance thresholds
                    if 'last_speed' in locals() and last_speed >= SPEED_THRESHOLD and distance >= DISTANCE_THRESHOLD:
                        total_distance += distance
                        print(last_speed)
                        print('this is somehow working')

                        # Update current x and y position using last known heading
                        rad_heading = math.radians(last_heading) if 'last_heading' in locals() else 0
                        delta_x = distance * math.cos(rad_heading)
                        delta_y = distance * math.sin(rad_heading)

                        current_x += delta_x
                        current_y += delta_y

                        # Display data
                        print(f"New position (lat, lon): ({latitude:.6f}, {longitude:.6f})")
                        print(f"Distance travelled in last update: {distance:.2f} meters")
                        print(f"Total distance travelled: {total_distance:.2f} meters")
                        print(f"Current x, y position: ({current_x:.2f}, {current_y:.2f})\n")
                    else:
                        print(f"Skipping update. Speed: {last_speed:.2f}, Distance: {distance:.2f}")

                # Update previous coordinates
                prev_lat, prev_lon = latitude, longitude

            elif nmea_sentence[0] == '$GPRMC' and len(nmea_sentence) >= 9:
                # Extract speed and heading from GPRMC
                last_speed = float(nmea_sentence[7]) * 1.852  # Convert knots to km/h
                last_heading = float(nmea_sentence[8])  # Heading in degrees

                # You can print speed and heading if needed
                print(f"Speed: {last_speed:.2f} km/h, Heading: {last_heading:.2f} degrees")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    client_socket.close()
    print("Connection closed.")
