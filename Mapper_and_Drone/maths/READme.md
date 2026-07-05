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

Quaternions.c evaluates input Euler angles and outputs Quaternions.