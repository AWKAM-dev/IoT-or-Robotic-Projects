#include <stdio.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <driver/gpio.h>

#define LED_PIN 5

// Dedicated FreeRTOS task for blinking the LED
void led_blink_task(void *pvParameters) {
    // Configure the GPIO pin inside the task
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
    int ON = 0;

    while (1) {
        ON = !ON;
        gpio_set_level(LED_PIN, ON);
        
        // Block the task for 1000ms, letting the CPU run other tasks/WDT
        vTaskDelay(pdMS_TO_TICKS(1000)); 
    }
}

void app_main(void) {
    // Create the task: Name, Stack size (2048 bytes), Params, Priority, Handle
    xTaskCreate(led_blink_task, "led_blink_task", 2048, NULL, 5, NULL);

    // app_main exits cleanly here; FreeRTOS handles the background execution
}
