#ifndef MOTOR_H
#define MOTOR_H

#include <stm32f4xx_hal_conf.h>

void resetIC();
void moveForward();
void moveBackward();
void brake();

#endif