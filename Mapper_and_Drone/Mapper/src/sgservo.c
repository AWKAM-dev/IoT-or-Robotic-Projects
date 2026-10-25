#include "../include/main.h"

void setServoAngle(){
    float angle = 0;
    
    //Linear mapping: 0 -> 1000us (1ms), 180 -> 2000 us (2ms)
    uint32_t pulse_ticks = 1000 + (uint32_t)((angle / 180.0f) * 1000.0f );

    //Update the Timer Capture/Compare Register dynamically
    __HAL_TIM_SetCompare(&htim2, TIM_CHANNEL_1, pulse_ticks);
}