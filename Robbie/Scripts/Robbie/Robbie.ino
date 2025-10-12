#include <Servo.h>
#include <math.h>

const bool runTest = false;
const bool runAgree = false;;
/*
recieve XY coordinates with unit length as CONV_FACT (defined later).
Convert XY to cm.
cm to Polar

For distances             GC
 <9 cm                    4.2
 26cm is limit            3.7

*/

//Units in cm
const float LB = 16.0;               // length LB (cm)
const float LT = 10.5;               // length LT (cm)
const float SC = 3.5;                // Stepper & servo clearance (cm)
float GC = 4;
const float step_trans = 3;

// const float DEG_TO_RAD = 0.017453292519943295f;
const float RAD_TO_DEGS = 57.29577951308232f;

const int stepsPerRev = 600;                 // stepper steps per revolution
const float STEPS_TO_DEG = 90.0/ (float)stepsPerRev;
const float DEG_TO_STEPS = (float)stepsPerRev / 90.0;

const int IN1 = 9;
const int IN2 = 11;
const int IN3 = 4;
const int IN4 = 10;

//Grid things
const float CONV_FACT = 1.75;
const float transX = 9.0f;
const float transY = 4.0f;

long stepCount = 0; //Track of Stepper rotation

//Matrix to manipulate stepper code better
const int writer[4][4] = {
  {1, 0, 0, 0},
  {0, 1, 0, 0},
  {0, 0, 1, 0},
  {0, 0, 0, 1}
};

//Datatype to store related value pairs
struct Pair {
  float first;
  float second;
};

//Function definitions

//Input
float floatFromSerial();

//Inverse Kinematics
Pair inverseKinematics(float distance);

//Movement
void smoothMove(int targetAngle, Servo input); //slow servo
void clockWise(int steps); //Rotate stepper clockwise
void antiClockWise(int steps); //other way
void moveArm(float botAng, float topAng, int degrees);
void toDump();
void waiter_pos();

Servo top, bottom, gripper;

void setup() {
  top.attach(5);
  top.write(0);
  delay(50);

  bottom.attach(3);
  bottom.write(0);
  delay(50);

  gripper.attach(6);
  gripper.write(0);
  delay(50);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  if(runTest){
  smoothMove(30, top);
  smoothMove(30, bottom);

  clockWise(90);
  delay(500);
  antiClockWise(90);

  smoothMove(0, bottom);
  smoothMove(0, top);

  gripper.write(120);
  delay(500);
  gripper.write(0);
  }


  Serial.begin(9600);
  Serial.println();
  Serial.println("Serial Communication online!");
}

void loop() {

  Serial.println("Enter the X coordinate: ");
  float usrX = floatFromSerial();
  Serial.print("X coord is: ");
  Serial.println(usrX);
  usrX -= transX;
  if(usrX > 31 || usrX < 0){
    Serial.println("Coords error! X coord out of bounds!");
    Serial.println("DONE");
    return;
  }

  Serial.println("Enter the Y coordinate: ");
  float usrY = floatFromSerial();
  Serial.print("Y coord is: ");
  Serial.println(usrY);
  usrY -= transY;
  if(usrY > 22 || usrY < 0){
    Serial.println("Coords error! Y coord out of bounds!");
    Serial.println("DONE");
    return;
  }

  usrX = usrX * CONV_FACT;
  usrY = usrY * CONV_FACT;

  Pair polarCoords = cartToPolar(usrX, usrY);

  // Serial.print("Enter the distance in CM: ");
  // Pair polarCoords;
  // polarCoords.first = floatFromSerial();
  // polarCoords.second = 0;

  Serial.print("Distance is: ");
  Serial.println(polarCoords.first);
  Serial.print("Angle is: ");
  Serial.println(polarCoords.second);

  if(polarCoords.first > 25.5 || polarCoords.first < 8){
    Serial.println("RISKY TERRITORY!");
    sayNo();
    Serial.println("DONE");
    return;
  }
  
  Pair servoRot = inverseK(polarCoords.first + step_trans - (polarCoords.first * 0.1));
  if(isnan(servoRot.first) || isnan(servoRot.second)){
    Serial.println("NAN ERROR!");
    sayNo();
    Serial.println("DONE");
    return;
  }
  Serial.print("Bottom Servo is at: ");
  Serial.println(servoRot.first);
  Serial.print("Top Servo is at: ");
  Serial.println(servoRot.second);

  if(runAgree){
  Serial.println("Carry on with process? Y/N: ");
  String usrAgree = "";
  while (usrAgree.length() == 0) {  
    if (Serial.available()) {
        usrAgree = Serial.readStringUntil('\n');
        usrAgree.trim();
    }
}

  if (usrAgree.equalsIgnoreCase("N")) {
      return; // abort
  } 
  else if (usrAgree.equalsIgnoreCase("Y")) {
      moveArm(servoRot.first, servoRot.second, (int)round(polarCoords.second)); 
      Serial.println("DONE");
  } 
  else {
      Serial.println("INVALID OPTION!");
  }
  }
  else {
  moveArm(servoRot.first, servoRot.second, (int)round(polarCoords.second)); 
  Serial.println("DONE");
  }

  //clear serial buffer
  while(Serial.available()) Serial.read();

}

Pair inverseKinematics(float distance){
  Pair output;
  float BF = sqrt(SC*SC + distance*distance);
  float EF = SC;
  float BE = distance;
  float DE = GC - EF;
  float BD = sqrt(BE*BE + DE*DE);

  float CBDNum = LT*LT - LB*LB - BD*BD;
  float CBDDen = -2 * BD * LB;
  Serial.print("CBD fraction is: ");
  Serial.println(CBDNum/CBDDen);

  float CBDRad = acos(CBDNum/CBDDen);
  float CBDDeg = CBDRad * RAD_TO_DEGS;
  
  float DBERad = atan2(DE, BE);
  float DBEDeg = DBERad * RAD_TO_DEGS;

  Serial.print("Bottom angle is: ");
  Serial.println(CBDDeg+DBEDeg);

  float CBENot =90 - CBDDeg - DBEDeg;

  float BCDNum = BD*BD - LT*LT - LB*LB;
  float BCDDen = -2 * LT * LB;
  Serial.print("BCD fraction is: ");
  Serial.println(BCDNum/BCDDen);

  float BCDRad = acos(BCDNum/BCDDen);
  float BCDDeg = BCDRad * RAD_TO_DEGS;

  Serial.print("Top angle is: ");
  Serial.println(BCDDeg);

  float BCDNot = 180 - BCDDeg;

  output.first = CBENot;
  output.second = BCDNot;

  return output;
}

Pair inverseK(float distance){
  Pair output;

  //Triangle ABE
  float BE = sqrt(SC*SC + distance*distance);
  float BEA = atan(distance/BE);
  float ABE = 90 - BEA;

  //Triangle BDE
  float BED = ABE;
  float BD = sqrt(BE*BE + GC*GC - 2*GC*BE*cos(BED));
  float EBD = acos((GC*GC - BD*BD - BE*BE)/(-2*BD*BE));

  //Triangle FBD
  float FBD = 90 - ((ABE+EBD)*RAD_TO_DEGS);

  //Triangle BCD
  float BCD = acos((BD*BD - LB*LB - LT*LT)/(-2*LB*LT));
  float CBD = acos((LT*LT - LB*LB - BD*BD)/(-2*LB*BD));

  //first is botServo.
  output.first = 90 - ((CBD+FBD)*RAD_TO_DEGS);
  output.second = 180 - BCD;
  
  return output;
  
}


void smoothMove(int targetAngle, Servo input){
    int currentAngle = input.read();

  //Decide if we have to move up or down the circle of movement
  int step = (currentAngle < targetAngle) ? 1 : -1;

  while(currentAngle != targetAngle){
    currentAngle += step;
    input.write(currentAngle);
    //Adjust delay to adjust servo speed
    delay(100);
    }
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

Pair cartToPolar (float x, float y){
  float r = sqrt(x*x + y*y);
  float theta = atan(y/x) * RAD_TO_DEGS;
  Pair output;
  output.first = r;
  output.second = theta;
  return output;
}

//Basic func to ask userInp then convert to float
float floatFromSerial() {
  while (Serial.available() == 0) {
    // Waiting period
  }
  String input = Serial.readStringUntil('\n');
  input.trim(); // Remove junk
  return input.toFloat();
}

void moveArm(float botAng, float topAng, int degrees){
  clockWise(degrees);
  smoothMove(topAng, top);
  gripper.write(180);
  smoothMove(botAng, bottom);

  smoothMove(0, gripper);
  toDump();
  waiter_pos();
}

void toDump(){
  int currDeg = stepCount * STEPS_TO_DEG;
  smoothMove(15, bottom);
  smoothMove(75,top);
  delay(500);
  clockWise(150 - currDeg);
  delay(500);
  gripper.write(180);
  delay(500);
}

void waiter_pos(){
  int currDeg = stepCount * STEPS_TO_DEG;
  smoothMove(0, top);
  smoothMove(0, bottom);
  gripper.write(0);
  antiClockWise(currDeg);
  delay(500);
}

void sayNo(){
  top.write(30);
  antiClockWise(45);
  gripper.write(120);
  for(int i = 0; i < 4; i++){
    clockWise(15);
    delay(500);
    antiClockWise(15);
    delay(500);
  }
  clockWise(90);
  waiter_pos();
}
