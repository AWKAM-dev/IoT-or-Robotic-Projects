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
}

void loop() {
  // put your main code here, to run repeatedly:

}
