'''import socket
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

        # Parse JSON if possible; otherwise ignore
        try:
            gps_data = json.loads(decoded_data)
            if not isinstance(gps_data, dict):  # Check if the parsed data is a dictionary
                continue  # Ignore if not a JSON object
        except json.JSONDecodeError:
            continue  # Ignore non-JSON messages like `8080`

        # Extract GPS data fields
        latitude = gps_data.get('latitude')
        longitude = gps_data.get('longitude')
        heading = gps_data.get('heading')  # in degrees
        speed_kmh = gps_data.get('speed')  # already in km/h

        # Calculate distance if previous location is available
        if prev_lat is not None and prev_lon is not None:
            distance = haversine(prev_lat, prev_lon, latitude, longitude)
            total_distance += distance

            # Convert distance and heading to x and y coordinates
            rad_heading = math.radians(heading)
            delta_x = distance * math.cos(rad_heading)
            delta_y = distance * math.sin(rad_heading)

            # Update current x and y position
            current_x += delta_x
            current_y += delta_y

            # Display data
            print(f"New position (lat, lon): ({latitude}, {longitude})")
            print(f"Distance travelled in last update: {distance:.2f} meters")
            print(f"Total distance travelled: {total_distance:.2f} meters")
            print(f"Current x, y position: ({current_x:.2f}, {current_y:.2f})")
            print(f"Speed: {speed_kmh:.2f} km/h")
            print(f"Heading: {heading:.2f} degrees\n")
        
        # Update previous coordinates
        prev_lat, prev_lon = latitude, longitude

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
                    total_distance += distance

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

                # Update previous coordinates
                prev_lat, prev_lon = latitude, longitude

            elif nmea_sentence[0] == '$GPRMC' and len(nmea_sentence) >= 9:
                # Extract speed and heading from GPRMC
                speed_kmh = float(nmea_sentence[7]) * 1.852  # Convert knots to km/h
                last_heading = float(nmea_sentence[8])  # Heading in degrees

                # You can print speed and heading if needed
                print(f"Speed: {speed_kmh:.2f} km/h, Heading: {last_heading:.2f} degrees")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    client_socket.close()
    print("Connection closed.")