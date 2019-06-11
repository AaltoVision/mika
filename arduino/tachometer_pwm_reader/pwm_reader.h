#include "Arduino.h"

class PWMReader {
  public:
  const static unsigned int Signals = 3;
  byte pins[Signals] = { 3, 4, 5 };

  unsigned int pw[Signals];
  bool values[Signals];
  unsigned long last_high[Signals] = {0};

  PWMReader();
  void valueChanged(int pin, bool value);
  void loop();
};

