#include <stm32f4xx_hal.h>

#define LED_PIN GPIO_PIN_13
#define LED_PORT GPIOC

void SysTick_Handler(void) {
    HAL_IncTick();
}

int main(void) {
    HAL_Init();

    __HAL_RCC_GPIOC_CLK_ENABLE(); //Give power to Port C logic gates

    //Declares a container variable named GPIO_InitStruct and initializes every field inside it to 0
    GPIO_InitTypeDef GPIO_InitStruct = {0}; 
    /*
    Before telling the hardware how to set up the pin, you need a temporary scratchpad in RAM to hold your choices. 
    Setting = {0} ensures that old garbage data left in that memory slot doesn't accidentally trigger weird settings.
    */

    //Specifies which physical pin on the chip you are talking about
    GPIO_InitStruct.Pin = LED_PIN;
    /*
    An STM32 chip has multiple ports (Port A, B, C) and each port has 16 individual pins (0 through 15). 
    This line tells the configuration function: "Apply the following settings specifically to Pin 13."
    */

    //Sets the pin to Push-Pull Output mode.
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP; //Output mode
    /*
    "Push-Pull" means the chip's internal circuitry can actively drive the pin voltage to either strong 3.3V (HIGH) or strong 0V (LOW). 
    This is the standard mode used for driving LEDs, buzzers, or logic gates.
    */
    
    //Disables any internal pull-up or pull-down resistors on the pin.
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    /*
    Since this pin is acting as an output (the chip is pushing voltage out to the LED), it doesn't need internal resistors to keep it stable. 
    Pull-up/down resistors are normally reserved for inputs (like buttons) so they don't float randomly when left unpressed.
    */

    //Sets the output transition speed (slew rate) to low.
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    /*
    When a pin switches from 0V to 3.3V, how fast should it transition? 
    Setting it to "Low" limits the transition speed, which reduces electrical noise, radio interference (EMI), and power consumption. 
    Since an LED doesn't care if it turns on in 2 nanoseconds or 50 nanoseconds, Low speed is more than enough.
    */
    
    //Once you finish filling out this "form," you pass its memory address into the initialization function:        
    HAL_GPIO_Init(LED_PORT, &GPIO_InitStruct);

    /*
    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_RESET); //High Voltage
    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_SET); //Low Voltage
    */

    while (1) {
        HAL_GPIO_TogglePin(LED_PORT, LED_PIN);
        HAL_Delay(500);
    }
}