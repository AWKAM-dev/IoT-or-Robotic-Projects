const int readPin = A0;

void setup() {
  pinMode(readPin, INPUT);
  Serial.begin(115200);

  Serial.println("Serial monitor active.");
}

void loop() {
  int readValue = analogRead(readPin);

  if(   readValue == 511){Serial.println("MIDDLE REACHED");}
  Serial.print("Input: ");
  Serial.println(analogRead(readPin));
  delay(50);
}
