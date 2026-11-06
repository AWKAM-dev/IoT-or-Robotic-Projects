#include "../include/main.h"

ADC_HandleTypeDef hadc1;

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

// val_pa6 = Read_ADC_Channel(ADC_CHANNEL_6);
//     val_pa7 = Read_ADC_Channel(ADC_CHANNEL_7);
//     val_pb0 = Read_ADC_Channel(ADC_CHANNEL_8);
//     val_pb1 = Read_ADC_Channel(ADC_CHANNEL_9);
