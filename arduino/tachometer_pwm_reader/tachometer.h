#include "Arduino.h"


class Tachometer {
  private:
  volatile unsigned int counter = 0;
  unsigned int toRPSMultiplier;

  public:
  volatile unsigned int rpm = 0;

  Tachometer(const unsigned int updateEvery);

  void incrementCounter();
  void updateRPM();
};

