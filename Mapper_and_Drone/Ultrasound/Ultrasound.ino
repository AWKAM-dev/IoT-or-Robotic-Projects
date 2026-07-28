const int trigPin = 8;
const int echoPin = 2;

//Angle is 15 degrees. 3 Ultrasound sensors should cover it

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  digitalWrite(trigPin, LOW);

  Serial.begin(115200);
  Serial.println("Serial monitor started");
}

void loop() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);


  int echo = pulseIn(echoPin, HIGH);
  echo = echo/50;
  if(echo != 0)Serial.println(echo);
}
