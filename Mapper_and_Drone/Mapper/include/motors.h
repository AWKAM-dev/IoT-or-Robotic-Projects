#ifndef MOTOR_H
#define MOTOR_H

#include <stm32f4xx_hal_conf.h>

void resetIC(char ICID);
void moveForward(char ICID);
void moveBackward(char ICID);
void brake(char ICID);

#endif