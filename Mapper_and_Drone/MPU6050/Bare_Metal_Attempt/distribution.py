import serial
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
ax, ay, az, gx, gy, gz
"""

#Configuring Serial port
PORT = "/dev/ttyACM0"
BAUDRATE = 115200

def parse_imu(serOut):
    """
    parse ax, ay, az, gx, gy, gz as list of floats
    """

    if not serOut:
        return None

    try:
        values = [float(val.strip()) for val in serOut.split(",")]

        if len(values) == 6:
            return values
        else:
            print(f"Did not recieve expected number of inputs. Recieved instead {len(values)}")
            return None

    except ValueError:
        print(f"Warning: Could not parse data string: {serOut}")
        return None
#End of parse_imu

data_list = []
SAMPLE_COUNT = 1000
print(f"Collecting information from {PORT} @ {BAUDRATE}. Maintain stable position of sensor")

try:
    with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
        print(f"Connected to {PORT}. Waiting for data...")
        time.sleep(2)

        while len(data_list) < SAMPLE_COUNT:
            #Read a line until a newline character is hit
            raw_data = ser.readline()

            #Only process if data was actually recieved
            if raw_data:
                #Decode bytes to string and strip whitespaces
                text_data = raw_data.decode('utf-8').strip()
                value_list = parse_imu(text_data)
                if value_list:
                    data_list.append(value_list)
                    if len(data_list) % 100 == 0:
                        print(f"Progress: {100*(len(data_list)/SAMPLE_COUNT)}")

except serial.SerialException as e:
    print(f"Error opening serial port as {e}")

except KeyboardInterrupt:
    print("\nProgramm stopped by user.")

