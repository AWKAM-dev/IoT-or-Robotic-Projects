#include "../include/main.h"
/*
A IN1 == PA4
A IN2 == PC14
A IN3 == PB5
A IN4 == PB4
A EN1 == PA2
A EN2 == PA3

B IN1 == PC13
B IN2 == PA11
B IN3 == PB14
B IN4 == PB13
B EN1 == PA5
B EN2 == PC15
*/

//Reset IC state
void resetIC(){
    //A
    HAL_GPIO_WritePin(GPIOA, L293D_A_IN1_Pin | L293D_A_EN1_Pin | L293D_A_EN2_Pin, RESET);
    HAL_GPIO_WritePin(GPIOB, L293D_A_IN3_Pin | L293D_A_IN4_Pin,RESET);
    HAL_GPIO_WritePin(GPIOC, L293D_A_IN2_Pin, RESET);

    //B
    HAL_GPIO_WritePin(GPIOA, L293D_B_IN2_Pin | L293D_B_EN1_Pin, RESET);
    HAL_GPIO_WritePin(GPIOB, L293D_B_IN3_Pin| L293D_B_IN4_Pin,RESET);
    HAL_GPIO_WritePin(GPIOC, L293D_B_IN1_Pin | L293D_B_EN2_Pin,RESET);
}


void moveForward(){
    //A
    HAL_GPIO_WritePin(GPIOA, L293D_A_EN1_Pin, SET);
    HAL_GPIO_WritePin(GPIOA, L293D_A_IN1_Pin, SET);
    HAL_GPIO_WritePin(GPIOC, L293D_A_IN2_Pin, RESET);

    HAL_GPIO_WritePin(GPIOA, L293D_A_EN2_Pin, SET);
    HAL_GPIO_WritePin(GPIOB, L293D_A_IN3_Pin, SET);
    HAL_GPIO_WritePin(GPIOB, L293D_A_IN4_Pin, RESET);

    //B
    HAL_GPIO_WritePin(GPIOA, L293D_B_EN1_Pin, SET);
    HAL_GPIO_WritePin(GPIOC, L293D_B_IN1_Pin, SET);
    HAL_GPIO_WritePin(GPIOA, L293D_B_IN2_Pin, RESET);

    HAL_GPIO_WritePin(GPIOC, L293D_B_EN2_Pin, SET);
    HAL_GPIO_WritePin(GPIOB, L293D_B_IN3_Pin, SET);
    HAL_GPIO_WritePin(GPIOB, L293D_B_IN4_Pin, RESET);
}

void moveBackward(){
    //A
    HAL_GPIO_WritePin(GPIOA, L293D_A_EN1_Pin, SET);
    HAL_GPIO_WritePin(GPIOA, L293D_A_IN1_Pin, RESET);
    HAL_GPIO_WritePin(GPIOC, L293D_A_IN2_Pin, SET);

    HAL_GPIO_WritePin(GPIOA, L293D_A_EN2_Pin, SET);
    HAL_GPIO_WritePin(GPIOB, L293D_A_IN3_Pin, RESET);
    HAL_GPIO_WritePin(GPIOB, L293D_A_IN4_Pin, SET);

    //B
    HAL_GPIO_WritePin(GPIOA, L293D_B_EN1_Pin, SET);
    HAL_GPIO_WritePin(GPIOC, L293D_B_IN1_Pin, RESET);
    HAL_GPIO_WritePin(GPIOA, L293D_B_IN2_Pin, SET);

    HAL_GPIO_WritePin(GPIOC, L293D_B_EN2_Pin, SET);
    HAL_GPIO_WritePin(GPIOB, L293D_B_IN3_Pin, RESET);
    HAL_GPIO_WritePin(GPIOB, L293D_B_IN4_Pin, SET);
}

void brake(){
    //A
    HAL_GPIO_WritePin(GPIOA, L293D_A_IN1_Pin | L293D_A_EN1_Pin | L293D_A_EN2_Pin, SET);
    HAL_GPIO_WritePin(GPIOB, L293D_A_IN3_Pin | L293D_A_IN4_Pin,SET);
    HAL_GPIO_WritePin(GPIOC, L293D_A_IN2_Pin, SET);

    //B
    HAL_GPIO_WritePin(GPIOA, L293D_B_IN2_Pin | L293D_B_EN1_Pin, SET);
    HAL_GPIO_WritePin(GPIOB, L293D_B_IN3_Pin| L293D_B_IN4_Pin,SET);
    HAL_GPIO_WritePin(GPIOC, L293D_B_IN1_Pin | L293D_B_EN2_Pin,SET);
}