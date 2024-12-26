import time
import json
from vectornav import Sensor, Registers


def track_location(sensor_port="COM30", duration=60):

    # initate sensor
    vs = Sensor()

    # connect to sensor
    vs.autoConnect(sensor_port)
    print(f"Connected to {sensor_port} at {vs.connectedBaudRate()}")

    # INS register assignment for GPS
    ins_sol_lla_register = Registers.InsSolLla()
    location_data = []

    # tracking for 60 secs
    t0 = time.time()
    while (time.time() - t0 < duration):
        vs.readRegister(ins_sol_lla_register)

        data_entry = {
            "gpsTow": ins_sol_lla_register.gpsTow,
            "timeGpsWeek": ins_sol_lla_register.timeGpsWeek,
            "insStatus": ins_sol_lla_register.insStatus.mode,  # Serialize only the mode attribute
            "yaw": ins_sol_lla_register.yaw,
            "pitch": ins_sol_lla_register.pitch,
            "roll": ins_sol_lla_register.roll,
            "posLat": ins_sol_lla_register.posLat,
            "posLon": ins_sol_lla_register.posLon,
            "posAlt": ins_sol_lla_register.posAlt,
            "velN": ins_sol_lla_register.velN,
            "velE": ins_sol_lla_register.velE,
            "velD": ins_sol_lla_register.velD,
            "attUncertainty": ins_sol_lla_register.attUncertainty,
            "posUncertainty": ins_sol_lla_register.posUncertainty,
            "velUncertainty": ins_sol_lla_register.velUncertainty
        }

        location_data.append(data_entry)

        print(data_entry)

        # slow down reading to see it
        time.sleep(1)

    # json data dump
    with open('location_data.json', 'w') as json_file:
        json.dump(location_data, json_file, indent=4)

    # manual says need to disconnect
    vs.disconnect()
    print("Sensor Disconnected")


#test
track_location(sensor_port="COM26", duration=60)