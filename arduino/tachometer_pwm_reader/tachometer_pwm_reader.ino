#include <util/atomic.h>
#include <Wire.h>
#include <MsTimer2.h>
#include "I2C_Anything.h"
#include "pwm_reader.h"
#include "tachometer.h"

const unsigned int MeasuringInterval = 100; // In milliseconds.
const byte HallSensor = 2; // Hall sensor at pin 2
const byte SlaveAddress = 0x06;

unsigned int sendBuffer[PWMReader::Signals + 1] = { 0 };

PWMReader reader;
Tachometer tachometer(MeasuringInterval);

void sendData() {
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    // First integer is RPM, the rest are the pulse widths in order.
    sendBuffer[0] = tachometer.rpm;
    for (uint8_t i=0; i < PWMReader::Signals; i++) {
      sendBuffer[i+1] = reader.pw[i];
    }
    I2C_writeAnything(sendBuffer);
  }
}

void hallInterrupt() {
  tachometer.incrementCounter();
}

void RPMCallback() {
  tachometer.updateRPM();
}

void setup() {

  // Setup pins for tachometer.
  pinMode(HallSensor, INPUT); //Sets hallsensor as input
  attachInterrupt(digitalPinToInterrupt(HallSensor), hallInterrupt, RISING); //Interrupts are called on Rise of Input

  // Setup PWM reader.
  MsTimer2::set(MeasuringInterval, RPMCallback);
  MsTimer2::start();

  Wire.begin(SlaveAddress);
  Wire.setClock(400000);
  Wire.onRequest(sendData);
}

void loop() {
  reader.loop();
}

