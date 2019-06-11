#include "pwm_reader.h"

PWMReader::PWMReader() {
  pinMode(pins[0], INPUT);
  pinMode(pins[1], INPUT);
}

void PWMReader::valueChanged(int pin, bool value) {
  // calculate pulse width.
  if (value == 1){
    last_high[pin] = micros();
  } else {
    // Pulse changed to low.
    pw[pin] = micros() - last_high[pin];
  }
  values[pin] = value;
}

void PWMReader::loop() {
  // Read values, check for change.
  for (int i = 0; i < Signals; i++){
    int value = digitalRead(pins[i]);
    if (value != values[i]){
      valueChanged(i, value);
    }
  }
}
