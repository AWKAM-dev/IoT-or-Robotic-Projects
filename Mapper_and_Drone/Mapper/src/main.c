/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
/* Includes ------------------------------------------------------------------*/
#include "../include/main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stm32f4xx_hal.h"
#include "stm32f4xx_hal_tim.h"
#include "stm32f4xx_hal_adc.h"
#include "stm32f4xx_hal_uart.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
TIM_HandleTypeDef htim2;
I2C_HandleTypeDef hi2c1;
ADC_HandleTypeDef hadc1;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define LIS3DH_I2C_ADDR (0x18 << 1) //7 bit address shifted for HAL
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;

UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_ADC1_Init(void);
static void MX_USART1_UART_Init(void);

/* USER CODE BEGIN PFP */
void I2C1_Init(void);
void LIS3DH_Init(void);

void ADC_Init(void);
uint32_t Read_ADC_Channel(uint32_t channel);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */
  I2C1_Init();
  LIS3DH_Init();

  ADC_Init();

  int16_t raw_accel[3];
  uint8_t reg_addr = 0x28 | 0x80; //0x28 is OUT_X_L, bit 7 high enables auto-increment
  
  uint32_t val_pa6 = 0;
  uint32_t val_pa7 = 0;
  uint32_t val_pb0 = 0;
  uint32_t val_pb1 = 0;
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ADC1_Init();
  MX_USART1_UART_Init();
  /* USER CODE BEGIN 2 */

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    //LIS3DH code
    //Read X, Y and Z accel data bytes (6 bytes total)
    HAL_I2C_Master_Transmit(&hi2c1, LIS3DH_I2C_ADDR, &reg_addr, 1, HAL_MAX_DELAY);
    HAL_I2C_Master_Recieve(&hi2c1, LIS3DH_I2C_ADDR, (uint8_t *)raw_accel, 6, HAL_MAX_DELAY);
    
    //Reading each potentiometer channel
    val_pa6 = Read_ADC_Channel(ADC_CHANNEL_6);
    val_pa7 = Read_ADC_Channel(ADC_CHANNEL_7);
    val_pb0 = Read_ADC_Channel(ADC_CHANNEL_8);
    val_pb1 = Read_ADC_Channel(ADC_CHANNEL_9);

    HAL_Delay(100);
    /* USER CODE END WHILE */

  }
  /* USER CODE BEGIN 3 */

  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 12;
  RCC_OscInitStruct.PLL.PLLN = 96;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }

  /** Enables the Clock Security System
  */
  HAL_RCC_EnableCSS();
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode = DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DMAContinuousRequests = DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_8;
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_3CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */
  __HAL_RCC_I2C1_CLK_ENABLE();
  __HAL_RCC_ADC1_CLK_ENABLE();
  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*
  Potentiometer_A_Sweep_GPIO_Port
  Potentiometer_B_Sweep_GPIO_Port
  Potentiometer_C_Sweep_GPIO_Port
  Potentiometer_D_Sweep_GPIO_Port
  */
  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, L293D_B_IN1_Pin|L293D_A_IN2_Pin|L293D_B_EN2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, L293D_A_EN1_Pin|L293D_A_EN2_Pin|L293D_A_IN1_Pin|L293D_B_EN1_Pin
                          |L293D_B_IN2_Pin|Ultrasound_B_Trig_Pin|Potentiometer_A_Sweep_Pin|Potentiometer_D_Sweep_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, Ultrasound_C_Trig_Pin|L293D_B_IN4_Pin|L293D_B_IN3_Pin|Ultrasound_A_Trig_Pin
                          |L293D_A_IN4_Pin|L293D_A_IN3_Pin|Potentiometer_B_Sweep_Pin|Potentiometer_C_Sweep_Pin, GPIO_PIN_RESET);

  //Configure Potentiometer Pins
  GPIO_InitStruct.Pin = Potentiometer_A_Sweep_Pin | Potentiometer_B_Sweep_Pin | Potentiometer_C_Sweep_Pin | Potentiometer_D_Sweep_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  //Configure the ADC Handle
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4; //Clock speed divider
  hadc1.Init.Resolution = ADC_RESOLUTION_12B; //0 to 4095
  hadc1.Init.ScanConvMode = DISABLE; //Single channel mode at a time
  hadc1.Init.ContinuousConvMode = DISABLE; //Don't free-run continously
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 1;
  HAL_ADC_Init(&hadc1);

  /*Configure GPIO pins : L293D_B_IN1_Pin L293D_A_IN2_Pin L293D_B_EN2_Pin */
  GPIO_InitStruct.Pin = L293D_B_IN1_Pin|L293D_A_IN2_Pin|L293D_B_EN2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
    
  /*Configure GPIO pin : Servo_PWM_Pin */
  GPIO_InitStruct.Pin = Servo_PWM_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF1_TIM2;
  HAL_GPIO_Init(Servo_PWM_GPIO_Port, &GPIO_InitStruct);

  //Configuring Timer base for 50Hz (Servo PWM) (1 us tick resolution)
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 100 - 1; //100MHz / 100 = 1 MHz (1 tick = 1 us)
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 20000 - 1; //20000 ticks = 20ms periods (50Hz) 
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  HAL_TIM_PWM_Init(&htim2);

  //Configure PWM Channel 1 Output parameters
  TIM_OC_InitTypeDef sConfigOC = {0};
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 1500;   //Default start position: 1500us (90)
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1);

  /*Configure GPIO pin : LIS3DH_INT1_Pin */
  GPIO_InitStruct.Pin = LIS3DH_INT1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING; //Trigger on rising edge
  GPIO_InitStruct.Pull = GPIO_PULLDOWN; 
  HAL_GPIO_Init(LIS3DH_INT1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : L293D_A_EN1_Pin L293D_A_EN2_Pin L293D_A_IN1_Pin L293D_B_EN1_Pin
                           L293D_B_IN2_Pin Ultrasound_B_Trig_Pin */
  GPIO_InitStruct.Pin = L293D_A_EN1_Pin|L293D_A_EN2_Pin|L293D_A_IN1_Pin|L293D_B_EN1_Pin
                          |L293D_B_IN2_Pin|Ultrasound_B_Trig_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PB2 PB12 */
  GPIO_InitStruct.Pin = GPIO_PIN_2|GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : Ultrasound_C_Trig_Pin L293D_B_IN4_Pin L293D_B_IN3_Pin Ultrasound_A_Trig_Pin
                           L293D_A_IN4_Pin L293D_A_IN3_Pin */
  GPIO_InitStruct.Pin = Ultrasound_C_Trig_Pin|L293D_B_IN4_Pin|L293D_B_IN3_Pin|Ultrasound_A_Trig_Pin
                          |L293D_A_IN4_Pin|L293D_A_IN3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : PA8 PA15 */
  GPIO_InitStruct.Pin = GPIO_PIN_8|GPIO_PIN_15;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : Ultrasound_C_Echo_Pin */
  GPIO_InitStruct.Pin = Ultrasound_C_Echo_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF1_TIM2;
  HAL_GPIO_Init(Ultrasound_C_Echo_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LIS3DH_SCL_Pin LIS3DH_SDA_Pin */
  GPIO_InitStruct.Pin = LIS3DH_SCL_Pin|LIS3DH_SDA_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_OD; //Must be open drain for I2C
  GPIO_InitStruct.Pull = GPIO_PULLUP; //Internal pullups required
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF4_I2C1; //AF4 maps PB6/7 to I2C1
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : Ultrasound_A_Echo_Pin Ultrasound_B_Echo_Pin */
  GPIO_InitStruct.Pin = Ultrasound_A_Echo_Pin|Ultrasound_B_Echo_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF2_TIM4;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

    //Enable and set Interrupt priority for INT1 in LIS3DH in NVIC
    HAL_NVIC_SetPriority(EXTI1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(EXTI1_IRQn);

    //Configure I2C1 Peripheral Handle
    hi2c1.Instance = I2C1;
    hi2c1.Init.ClockSpeed = 100000;
    hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
    hi2c1.Init.OwnAddress1 = 0;
    hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c1.Init.OwnAddress2 = 0;
    hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;

    HAL_I2C_Init(&hi2c1);


  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
void LIS3DH_Init(void){
  uint8_t config_data[2];

  //Enable all axes and set output data rate to 50Hz (CTRL_REG1 = 0x47)
  config_data[0] = 0x20;
  config_data[1] = 0x47;
  HAL_I2C_Master_Transmit(&hi2c1, LIS3DH_I2C_ADDR, config_data, 2, HAL_MAX_DELAY);

  //Set full-scale range to +/- 2g (CTRL_REG4 = 0x08)
  config_data[0] = 0x23;
  config_data[1]= 0x08;
  HAL_I2C_Master_Transmit(&hi2c1, LIS3DH_I2C_ADDR, config_data, 2, HAL_MAX_DELAY);
}

uint32_t Read_ADC_Channel(uint32_t channel){
  ADC_ChannelConfTypeDef sConfig = {0};
  sConfig.Channel = channel;
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_3CYCLES;

  //Apply channel selection to ADC
  HAL_ADC_ConfigChannel(&hadc1, &sConfig);

  //Start conversion, wait for result, stop
  HAL_ADC_Stop(&hadc1);
  if (HAL_ADC_PollForConversion(&hadc1, 100) == HAL_OK) {
      uint32_t adc_value = HAL_ADC_GetValue(&hadc1);
      HAL_ADC_Stop(&hadc1);
      return adc_value;
  }

  HAL_ADC_Stop(&hadc1);
  return 0; //Return 0 if timeout occurs
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
