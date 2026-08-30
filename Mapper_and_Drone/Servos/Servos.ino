#include <Servo.h>

Servo myServo;
int servoPin = 3;
float angIncr = 11.67;

void setup() {
  myServo.attach(servoPin);
  pinMode(LED_BUILTIN, OUTPUT);

  myServo.writeMicroseconds(500);

  digitalWrite(LED_BUILTIN, HIGH);
  delay(2);
  digitalWrite(LED_BUILTIN, LOW);
}

void loop() {
  for(int i = 0; i < 180; i++){
    myServo.write(i);
    delay(15);
  }

  for(int i = 180; i > -1; i--){
    myServo.write(i);
    delay(15);
  }

//  for(float i = 500; i < 2600; i += angIncr){
//    myServo.writeMicroseconds(i);
//    delay(100);
//  }

  digitalWrite(LED_BUILTIN, HIGH);
  delay(200);
  digitalWrite(LED_BUILTIN, LOW);
}
