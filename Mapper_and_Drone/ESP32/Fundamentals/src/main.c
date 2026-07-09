#include <stdio.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_log.h>

static const char *TAG = "RTOS_DEMO";

//Define the task function
void my_first_task(void *pvParameters) {
    //Either run in infinite loop or delete itself. Never return
    while(1) {
        ESP_LOGI(TAG ,"Hello from my first task!\n");

        //vTaskDelay expects ticks. pdMS_TO_TICKS converts milliseconds to ticks. 1000ms = 1s
        vTaskDelay(pdMS_TO_TICKS(500));
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
        5, //priority
        NULL //task handle
    );

    while(1){
        //Main app is a task in of itself. So it should not return either
        ESP_LOGI(TAG, "Hello from main task!\n");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}