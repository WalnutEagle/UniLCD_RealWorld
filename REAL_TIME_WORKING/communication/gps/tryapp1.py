'''import socket

def start_udp_server():
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Bind the socket to the address and port
    server_address = ('0.0.0.0', 8083)  # Listening on all interfaces
    sock.bind(server_address)

    print(f"Listening for UDP packets on {server_address[0]}:{server_address[1]}...")

    while True:
        # Wait for a message
        data, address = sock.recvfrom(4096)  # Buffer size is 4096 bytes
        print(f"Received {data} from {address}")

if __name__ == "__main__":
    start_udp_server()

    
'''

import socket

def parse_nmea_data(data):
    # Decode the byte data to string
    nmea_sentence = data.decode('utf-8').strip()

    latitude, longitude, speed_kmh, bearing = None, None, None, None

    if nmea_sentence.startswith('$GNRMC'):
        parts = nmea_sentence.split(',')
        if len(parts) > 5:
            latitude = parts[3] + parts[4]  # Latitude and hemisphere
            longitude = parts[5] + parts[6]  # Longitude and hemisphere
            speed_knots = float(parts[7])  # Speed in knots
            speed_kmh = speed_knots * 1.852  # Convert knots to km/h
            bearing = parts[8]  # Course over ground

    return latitude, longitude, speed_kmh, bearing

def start_udp_server():
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the address and port
    server_address = ('0.0.0.0', 8083)  # Listening on all interfaces
    sock.bind(server_address)

    print(f"Listening for UDP packets on {server_address[0]}:{server_address[1]}...")

    while True:
        # Wait for a message
        data, address = sock.recvfrom(4096)  # Buffer size is 4096 bytes
        print(f"Received {data} from {address}")

        # Parse the NMEA data
        latitude, longitude, speed_kmh, bearing = parse_nmea_data(data)

        if latitude and longitude:
            print(f"Latitude: {latitude}, Longitude: {longitude}, Speed: {speed_kmh:.2f} km/h, Bearing: {bearing}")

if __name__ == "__main__":
    start_udp_server()
