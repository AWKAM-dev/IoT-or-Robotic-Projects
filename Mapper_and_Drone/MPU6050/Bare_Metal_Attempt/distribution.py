import serial
import time

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
        values = [float(val.strip()) for val in serOut.split("\n")]

        if len(values) == 6:
            return values
        else:
            print(f"Did not recieve expected number of inputs. Recieved instead {len(values)}")
            return None

    except ValueError:
        print(f"Warning: Could not parse data string: {serOut}")
        return None
#End of parse_imu


try:
    with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
        print(f"Connected to {PORT}. Waiting for data...")
        time.sleep(2)

        while True:
            #Read a line until a newline character is hit
            raw_data = ser.readline()

            #Only process if data was actually recieved
            if raw_data:
                #Decode bytes to string and strip whitespaces
                text_data = raw_data.decode('utf-8').strip()
                getArray(text_data)
                print(f"Recieved {text_data}")

except serial.SerialException as e:
    print(f"Error opening serial port as {e}")

except KeyboardInterrupt:
    print("\nProgramm stopped by user.")