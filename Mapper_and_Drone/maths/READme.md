📁 MyDroneFlightController/
│
├── 📁 include/                  # Global include files for main app
│   └── main.h
│
├── 📁 src/                      # Your main application loop
│   ├── main.c
│   └── CMakeLists.txt
│
├── 📁 lib/                      # 👈 YOUR SANDBOX LIBRARIES LIVE HERE
│   ├── 📁 imu_driver/
│   │   ├── 📁 include/
│   │   │   └── imu_driver.h
│   │   ├── 📁 src/
│   │   │   └── imu_driver.c
│   │   └── CMakeLists.txt       # 👈 Component registration
│   │
│   └── 📁 sensor_fusion/
│       ├── 📁 include/
│       │   └── sensor_fusion.h
│       ├── 📁 src/
│       │   └── sensor_fusion.c
│       └── CMakeLists.txt
│
└── 📄 platformio.ini            # Project configuration

# lib/imu_driver/CMakeLists.txt

idf_component_register(
    SRCS "src/imu_driver.c"
    INCLUDE_DIRS "include"
    REQUIRES driver            # Tells ESP-IDF to link native I2C/GPIO drivers
)

Quaternions.c evaluates input Euler angles and outputs Quaternions.