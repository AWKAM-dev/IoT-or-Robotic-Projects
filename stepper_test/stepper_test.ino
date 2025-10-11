
const int stepsPerRev = 600;                 // stepper steps per revolution


const int IN1 = 9;
const int IN2 = 8;
const int IN3 = 4;
const int IN4 = 2;

long stepCount = 0; //Track of Stepper rotation
const float STEPS_TO_DEG = 90.0/ (float)stepsPerRev;
const float DEG_TO_STEPS = (float)stepsPerRev / 90.0;

//Matrix to manipulate stepper code better
const int writer[4][4] = {
  {1, 0, 0, 0},
  {0, 1, 0, 0},
  {0, 0, 1, 0},
  {0, 0, 0, 1}
};


void clockWise(int steps); //Rotate stepper clockwise
void antiClockWise(int steps); //other way

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
}

void loop() {
  clockWise(45);
  delay(1000);
  antiClockWise(45);
  delay(1000);
  

}

void clockWise(int steps) {
  steps = (int)round(steps * DEG_TO_STEPS);
  //Check if stepper is already past 120 degrees.
  if(stepCount+steps > DEG_TO_STEPS * 180){
    Serial.println("Stepper out of bounds!");
    return;
  }

  //Add onto stepCount
  stepCount += steps;
  while (steps > 0) {
    for (int k = 0; k < 4; k++) {
      //Loop through writer matrix to turn on IN pins in order
      digitalWrite(IN1, writer[k][0]);
      digitalWrite(IN2, writer[k][1]);
      digitalWrite(IN3, writer[k][2]);
      digitalWrite(IN4, writer[k][3]);
      delay(3); // Adjust as needed
      steps--;
    }
  }
  
  Serial.print("Stepper is at: ");
  Serial.println(STEPS_TO_DEG * stepCount);

}

void antiClockWise(int steps) {
  steps = (int)round(steps * DEG_TO_STEPS);
  //Check whether given input surpasses traversed steps
  if(stepCount - steps < -90 * DEG_TO_STEPS){
    Serial.println("Stepper out of bounds!");
    return;
  }

  stepCount -= steps;
  while (steps > 0) {
    for (int k = 3; k >= 0; k--) {
      digitalWrite(IN1, writer[k][0]);
      digitalWrite(IN2, writer[k][1]);
      digitalWrite(IN3, writer[k][2]);
      digitalWrite(IN4, writer[k][3]);
      delay(3);
      steps--;
    }
  }
  Serial.print("Stepper is at: ");
  Serial.println(STEPS_TO_DEG * stepCount);
}
