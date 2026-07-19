#define MPU6050_ADDR 0x68
#define TOL_VAL 1

//RAW hardware helper functions for TWI (I2C) handling
void i2c_init() {
  TWBR = 72; //Set SCL Frequency to 100 kHz (CPU Freq / (16 + 2*TWBR*Prescaler)
  TWSR = 0x00; //Prescalar - 1
}

void i2c_start() {
  TWCR = (1 << TWINT) | (1 << TWSTA) | (1 << TWEN); //Clear flag, generate START, enable TWI
  while (!(TWCR & (1 << TWINT)));
}

void i2c_stop() {
  TWCR = (1 << TWINT) | (1 << TWSTO) | (1 << TWEN); //Clear flag, generate STOP, enable TWI
  while(!(TWCR & (1 << TWSTO))); //Wait until stop condition is executed
}

void i2c_write(uint8_t data) {
  TWDR = data;  //Load byte into data register
  TWCR = (1 << TWINT) | (1 << TWEN);  //Clear flag to initiate transmission.
  while (!(TWCR & (1 << TWINT))); //Wait until transmission finished
}

uint8_t i2c_read_ack() {
  TWCR = (1 << TWINT) | (1 << TWEN) | (1 << TWEA); //Clear flag, enable TWI, enable Acknowledge (ACK)
  while (!(TWCR & (1 << TWINT))); //Wait for data
  return TWDR; //Returh read byte
}

uint8_t i2c_read_nack() {
  TWCR = (1 << TWINT) | (1 << TWEN); //Clear flag, enable TWI, enable ACK
  while(!(TWCR & (1 << TWINT))); //Wait for data
  return TWDR; //Return read byte
}

void setup() {
  Serial.begin(115200);
  i2c_init();

  //Wake up MPU6050 out of sleep mode by writing 0x00 to PWR_MGMT_1 (Register 0x6B)
  i2c_start();
  i2c_write((MPU6050_ADDR << 1) | 0); //Address + Write bit
  i2c_write(0x6B); //PWR_MGMT_1 register address
  i2c_write(0x00); //Set to 0 to wake it up
  i2c_stop();

  Serial.println("Baremetal TWI initialization complete!");
}

void loop() {
  uint8_t rawData[14];

  //Point to the starting register address: ACCEL_XOUT_H (0x3B)
  i2c_start();
  i2c_write((MPU6050_ADDR << 1 | 0)); //Write address
  i2c_write(0x3B);  //Register 0x3B

  //Re-start to flip direction into Read mode
  i2c_start();
  i2c_write((MPU6050_ADDR << 1) | 1); //Read Address

  //Burst read all 14 registers sequentially

  for(int i = 0; i < 13; i++){
    rawData[i] = i2c_read_ack(); //Pull data and ask for more
  }
  rawData[13] = i2c_read_nack(); //Pull final byte, signal stop (NACK)
  i2c_stop(); //Free up bus
  
  //Reassemble the high and low 8-bit pieces into signed 16-bit variables
  int16_t raw_accel_x = (rawData[0] << 8) | rawData[1];
  int16_t raw_accel_y = (rawData[2] << 8) | rawData[3];
  int16_t raw_accel_z = (rawData[4] << 8) | rawData[5];

  int16_t raw_gyro_x = (rawData[8] << 8) | rawData[9];
  int16_t raw_gyro_y = (rawData[10] << 8) | rawData[11];
  int16_t raw_gyro_z = (rawData[12] << 8 | rawData[13]);

  // 2. Apply conversion constants to find actual physical units
  // Accelerometer conversion: Raw / 16384.0 * 9.80665
  float ax = (float)raw_accel_x / 16384.0 * 9.80665;
  float ay = (float)raw_accel_y / 16384.0 * 9.80665;
  float az = (float)raw_accel_z / 16384.0 * 9.80665;

  // Gyroscope conversion: Raw / 131.0 * (PI / 180.0)
  float gx = (float)raw_gyro_x / 131.0 * 0.01745329251;
  float gy = (float)raw_gyro_y / 131.0 * 0.01745329251;
  float gz = (float)raw_gyro_z / 131.0 * 0.01745329251;

  //if( ((ax >= TOL_VAL) || (ay >= TOL_VAL) | (az >= TOL_VAL) || (gx >= TOL_VAL) || (gy >= TOL_VAL) || (gz >= TOL_VAL + 9.8) || (ax <= -TOL_VAL) || (ay <= -TOL_VAL) || (az <= -TOL_VAL) || (gx <= -TOL_VAL) || (gy <= -TOL_VAL) || (gz <= -TOL_VAL))){

  //Print raw values to test communication stability
  Serial.println("AX: "); Serial.println(ax-2.02);
  Serial.println("AY: "); Serial.println(ay);
  Serial.println("AZ: "); Serial.println(az-10.10);
  Serial.println("GX: "); Serial.println(gx);
  Serial.println("GY: "); Serial.println(gy);
  Serial.println("GZ: "); Serial.println(gz);

  //}

  delay(500); //Fast cycle loop close to the 5ms drone loop
}
