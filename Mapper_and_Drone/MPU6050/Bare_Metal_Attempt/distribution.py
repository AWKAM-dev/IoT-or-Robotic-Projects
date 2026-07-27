import serial
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
ax, ay, az, gx, gy, gz

Most of the code here is AI generated as I am yet to familiriaze myself with Python and all its vast libraries
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
SAMPLE_COUNT = 10000
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

#Convert to pandas datafram
columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
df = pd.DataFrame(data_list, columns=columns)

#Display Summary Statistics
print("\n--- Sensor Noise Statistics ---")
print(df.describe().T[['mean', 'std', 'min', 'max']])

#Plot distributions using seaborn and matplotlib
fig, axes = plt.subplots(2, 3, figsize=(14,8))
fig.suptitle("MPU6050 Noise Distribution (Static Sensor)", fontsize=16)

for i, col in enumerate(['ax', 'ay', 'az']):
    sns.histplot(df[col], kde=True, ax=axes[0, i], color='tab:blue', bins=30)
    axes[0,i].set_title(f"Accel {col.upper()} (m/s2)")
    axes[0,i].set_xlabel("Value")
    axes[0,i].set_ylabel("Count")

for i, col in enumerate(['gx', 'gy', 'gz']):
    sns.histplot(df[col], kde=True, ax=axes[1, i], color='tab:orange', bins=30)
    axes[1, i].set_title(f"Gyro {col.upper()} (rad/s)")
    axes[1, i].set_xlabel("Value")
    axes[1, i].set_ylabel("Count")

plt.tight_layout()

plt.savefig("mpu6050_distribution.png", dpi=300, bbox_inches='tight')
print("Plot saved successfully")

plt.show()

"""
27 July 2026
--- Sensor Noise Statistics ---
         mean       std    min    max
ax   0.552248  0.032744   0.43   0.72
ay   0.070174  0.032617  -0.05   0.19
az  10.220433  0.049666  10.03  10.40
gx  -0.016268  0.004837  -0.02  -0.01
gy  -0.006242  0.004844  -0.01  -0.00
gz  -0.003945  0.004904  -0.02   0.01
"""