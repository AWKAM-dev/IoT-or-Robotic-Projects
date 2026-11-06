#include "../include/main.h"

I2C_HandleTypeDef hi2c1;

#define LIS3DH_I2C_ADDR (0x18 << 1) //7 bit address shifted for HAL

void LIS3DH_Init(void);

//Initialize LIS3DH
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