#include <stdio.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_log.h>
#include <driver/gpio.h>

#define LED_PIN 2

//Apparently ESPIDF doesn't prefer printfs.
static const char *TAG = "RTOS_DEMO";

//Define the task function
void my_first_task(void *pvParameters) {
    //Either run in infinite loop or delete itself. Never return
    while(1) {
        ESP_LOGI(TAG ,"Hello from my first task!\n");

        //vTaskDelay expects ticks. pdMS_TO_TICKS converts milliseconds to ticks. 1000ms = 1s
        vTaskDelay(pdMS_TO_TICKS(200));
    }
}

void led_blink_task(void *pvParameters){
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);

    int ON = 1;
    ESP_LOGI(TAG, "GPIO Initialization for GPIO2 is finished\n");

    while(1){
        ON = !ON;
        gpio_set_level(LED_PIN, ON);

        ESP_LOGI(TAG, "GPIO 2 at %s", ON ? "ON" : "OFF");

        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

void higher_led(void *pvParameters) {
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);

    int ON = 1;
    int count = 0;
    ESP_LOGI(TAG, "Higher LED task is initiialized\n");

    while(1){
        ON = !ON;
        count = !count;
        if (count){ gpio_set_level(LED_PIN, ON);}
        ESP_LOGI(TAG, "Higher LED is: %s", ON ? "ON" : "OFF");
        vTaskDelay(pdMS_TO_TICKS(300));
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "Starting the main application\n");

    //Task creation
    //Using standard method. ESPIDF will automatically allocate a core
    xTaskCreate(
        my_first_task, //Function that implements the task
        "MyFirstTask", //Text name for task (debugging guide)
        2048, //Stack size
        NULL, //task parameter
        3, //priority
        NULL //task handle
    );

    xTaskCreate(
        led_blink_task, //Function that implements the task
        "LED", //Text name for task (debugging guide)
        2048, //Stack size
        NULL, //task parameter
        4, //priority
        NULL //task handle
    );

    xTaskCreate(
        higher_led,
        "HighLED",
        2048,
        NULL,
        5,
        NULL
    );

    while(1){
        //Main app is a task in of itself. So it should not return either
        ESP_LOGI(TAG, "Hello from main task!\n");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}