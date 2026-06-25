#include <Wire.h>
#include <Adafruit_LIS3DH.h>
#include <Adafruit_Sensor.h>

Adafruit_LIS3DH lis = Adafruit_LIS3DH();

void setup() {
  Serial.begin(115200);
  while(!Serial) delay(10); //Wait for Serial monitor to load up
}

void loop() {
  // put your main code here, to run repeatedly:

}
