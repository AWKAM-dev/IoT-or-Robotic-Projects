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
void resetIC(char ICID){
    if(ICID == "A"){
    HAL_GPIO_WritePin(GPIOA, L293D_A_IN1_Pin | L293D_A_EN1_Pin | L293D_A_EN2_Pin, RESET);
    HAL_GPIO_WritePin(GPIOB, L293D_A_IN3_Pin | L293D_A_IN4_Pin,RESET);
    HAL_GPIO_WritePin(GPIOC, L293D_A_IN2_Pin, RESET);
    }

    if(ICID == "B"){
        HAL_GPIO_WritePin(GPIOA, L293D_B_IN2_Pin | L293D_B_EN1_Pin, RESET);
        HAL_GPIO_WritePin(GPIOB, L293D_B_IN3_Pin| L293D_B_IN4_Pin,RESET);
        HAL_GPIO_WritePin(GPIOC, L293D_B_IN1_Pin | L293D_B_EN2_Pin,RESET);
    }
}

void moveForward(char ICID){
    if(ICID == "A"){

    }

    if(ICID == "B"){
        
    }
}

void moveBackward(char ICID){
    if(ICID == "A"){

    }

    if(ICID == "B"){
        
    }
}

void brake(char ICID){
    //A
    HAL_GPIO_WritePin(GPIOA, L293D_A_IN1_Pin | L293D_A_EN1_Pin | L293D_A_EN2_Pin, SET);
    HAL_GPIO_WritePin(GPIOB, L293D_A_IN3_Pin | L293D_A_IN4_Pin,SET);
    HAL_GPIO_WritePin(GPIOC, L293D_A_IN2_Pin, SET);

    //B
    HAL_GPIO_WritePin(GPIOA, L293D_B_IN2_Pin | L293D_B_EN1_Pin, SET);
    HAL_GPIO_WritePin(GPIOB, L293D_B_IN3_Pin| L293D_B_IN4_Pin,SET);
    HAL_GPIO_WritePin(GPIOC, L293D_B_IN1_Pin | L293D_B_EN2_Pin,SET);
}