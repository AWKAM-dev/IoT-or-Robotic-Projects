import serial
import time

#Configuring Serial port
PORT = "/dev/ttyACM0"
BAUDRATE = 9600

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
                text_data = raw_data.replace("'", "").strip()
                print(f"Recieved {text_data}")

except serial.SerialException as e:
    print(f"Error opening serial port as {e}")

except KeyboardInterrupt:
    print("\nProgramm stopped by user.")