import vectornav
from vectornav import Sensor, Registers

def getgnssdata():
    # Connect to the GNSS sensor via COM26 and set the baud rate
    vs = Sensor()
    vs.autoConnect("COM26")
    vs.changeBaudRate(Sensor.BaudRate.Baud115200)

    try:
        # Read the Model Register as an example
        modelRegister = Registers.Model()
        vs.readRegister(modelRegister)
        print("Sensor Model:", modelRegister.model)

        # Now let's read other important GNSS registers

        # Read GNSS Basic Configuration (Example)
        gnss_basic_config = Registers.GnssBasicConfig()
        vs.readRegister(gnss_basic_config)
        print("GNSS Basic Config:", gnss_basic_config)

        # Read GNSS Solution in LLA (Latitude, Longitude, Altitude)
        gnss_sol_lla = Registers.GnssSolLla()
        vs.readRegister(gnss_sol_lla)
        print("GNSS Solution LLA:", gnss_sol_lla)

        # Read GNSS Solution in ECEF (Earth-Centered, Earth-Fixed) coordinates
        gnss_sol_ecef = Registers.GnssSolEcef()
        vs.readRegister(gnss_sol_ecef)
        print("GNSS Solution ECEF:", gnss_sol_ecef)

        # Read GNSS 2nd Solution in LLA (if available)
        gnss2_sol_lla = Registers.Gnss2SolLla()
        vs.readRegister(gnss2_sol_lla)
        print("GNSS 2nd Solution LLA:", gnss2_sol_lla)

        # Read GNSS System Configuration
        gnss_system_config = Registers.GnssSystemConfig()
        vs.readRegister(gnss_system_config)
        print("GNSS System Config:", gnss_system_config)

        # Read GNSS Synchronization Configuration
        gnss_sync_config = Registers.GnssSyncConfig()
        vs.readRegister(gnss_sync_config)
        print("GNSS Sync Config:", gnss_sync_config)

        # Read GNSS Compass Data (Health Status)
        gnss_compass_status = Registers.GnssCompassSignalHealthStatus()
        vs.readRegister(gnss_compass_status)
        print("GNSS Compass Health Status:", gnss_compass_status)

    except Exception as e:
        print("An error occurred:", e)

    finally:
        # Close the sensor connection after use
        vs.disconnect()

# Call the function to get GNSS data
getgnssdata()
