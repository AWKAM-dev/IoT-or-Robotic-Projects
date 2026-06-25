#include <Wire.h>
#include <Adafruit_LIS3DH.h>
#include <Adafruit_Sensor.h>

Adafruit_LIS3DH lis = Adafruit_LIS3DH();

void setup() {
  Serial.begin(115200);
  while(!Serial) delay(10); //Wait for Serial monitor to load up

  Serial.println("LIS3DH Accelerometer Test");

  //Initialize sensor with the default I2C address (0x18). Ensure GND to SDO to lock down 0x18
  if(!lis.begin(0x18)){
    Serial.println("Couldn't find a valid LIS3DH sensor, check wiring!");
    while(1) delay(10);
  }
  Serial.println("LIS3DH found!");

  //Set sensor measurement range (LIS3DH_RANGE_2_G, LIS3DH_RANGE_4_G, LIS3DH_RANGE_8_G, LIS3DH_RANGE_16_G)
  lis.setRange(LIS3DH_RANGE_2_G);

  Serial.print("Range set to: ");
  Serial.print(2 << lis.getRange());
  Serial.println("G");

  //Set data rate: Lower data rate = lower power consumption
  //Options: LIS3DH_DATARATE_1_HZ - LIS3DH_DATARATE_400_HZ, or LIS3DH_DATARATE_POWERDOWN
  lis.setDataRate(LIS3DH_DATARATE_50_HZ);
}

void loop() {
  /* 1. Read raw sensor data (registers) */
  lis.read();

  /* 2. Read standardized physics data (m/s^2) using Adafruit unified sensor framework */
  sensors_event_t event;
  lis.getEvent(&event);

  /* 3. Display the results */
  Serial.print("X: ");
  Serial.print(event.acceleration.x);
  Serial.print(" m/s^2");

  Serial.print("Y: ");
  Serial.print(event.acceleration.y);
  Serial.print(" m/s^2");

  Serial.print("Z: ");
  Serial.print(event.acceleration.z);
  Serial.print(" m/s^2");

  /* Optional: Print raw ADC readings if you have anything wired to ADC1/2/3 */
  // Serial.print("ADC1: "); Serial.println(lis.readADC(1));

  delay(200);
}
