/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define L293D_B_IN1_Pin GPIO_PIN_13
#define L293D_B_IN1_GPIO_Port GPIOC
#define L293D_A_IN2_Pin GPIO_PIN_14
#define L293D_A_IN2_GPIO_Port GPIOC
#define L293D_B_EN2_Pin GPIO_PIN_15
#define L293D_B_EN2_GPIO_Port GPIOC
#define Servo_PWM_Pin GPIO_PIN_0
#define Servo_PWM_GPIO_Port GPIOA
#define LIS3DH_INT1_Pin GPIO_PIN_1
#define LIS3DH_INT1_GPIO_Port GPIOA
#define L293D_A_EN1_Pin GPIO_PIN_2
#define L293D_A_EN1_GPIO_Port GPIOA
#define L293D_A_EN2_Pin GPIO_PIN_3
#define L293D_A_EN2_GPIO_Port GPIOA
#define L293D_A_IN1_Pin GPIO_PIN_4
#define L293D_A_IN1_GPIO_Port GPIOA
#define L293D_B_EN1_Pin GPIO_PIN_5
#define L293D_B_EN1_GPIO_Port GPIOA
#define Potentiometer_D_Sweep_Pin GPIO_PIN_6
#define Potentiometer_D_Sweep_GPIO_Port GPIOA
#define Potentiometer_A_Sweep_Pin GPIO_PIN_7
#define Potentiometer_A_Sweep_GPIO_Port GPIOA
#define Potentiometer_C_Sweep_Pin GPIO_PIN_0
#define Potentiometer_C_Sweep_GPIO_Port GPIOB
#define Potentiometer_B_Sweep_Pin GPIO_PIN_1
#define Potentiometer_B_Sweep_GPIO_Port GPIOB
#define Ultrasound_C_Trig_Pin GPIO_PIN_10
#define Ultrasound_C_Trig_GPIO_Port GPIOB
#define L293D_B_IN4_Pin GPIO_PIN_13
#define L293D_B_IN4_GPIO_Port GPIOB
#define L293D_B_IN3_Pin GPIO_PIN_14
#define L293D_B_IN3_GPIO_Port GPIOB
#define Ultrasound_A_Trig_Pin GPIO_PIN_15
#define Ultrasound_A_Trig_GPIO_Port GPIOB
#define ESP32_CAM_RX_Pin GPIO_PIN_9
#define ESP32_CAM_RX_GPIO_Port GPIOA
#define ESP32_CAM_TX_Pin GPIO_PIN_10
#define ESP32_CAM_TX_GPIO_Port GPIOA
#define L293D_B_IN2_Pin GPIO_PIN_11
#define L293D_B_IN2_GPIO_Port GPIOA
#define Ultrasound_B_Trig_Pin GPIO_PIN_12
#define Ultrasound_B_Trig_GPIO_Port GPIOA
#define Ultrasound_C_Echo_Pin GPIO_PIN_3
#define Ultrasound_C_Echo_GPIO_Port GPIOB
#define L293D_A_IN4_Pin GPIO_PIN_4
#define L293D_A_IN4_GPIO_Port GPIOB
#define L293D_A_IN3_Pin GPIO_PIN_5
#define L293D_A_IN3_GPIO_Port GPIOB
#define LIS3DH_SCL_Pin GPIO_PIN_6
#define LIS3DH_SCL_GPIO_Port GPIOB
#define LIS3DH_SDA_Pin GPIO_PIN_7
#define LIS3DH_SDA_GPIO_Port GPIOB
#define Ultrasound_A_Echo_Pin GPIO_PIN_8
#define Ultrasound_A_Echo_GPIO_Port GPIOB
#define Ultrasound_B_Echo_Pin GPIO_PIN_9
#define Ultrasound_B_Echo_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
