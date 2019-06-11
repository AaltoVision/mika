#include <util/atomic.h>
#include "tachometer.h"

Tachometer::Tachometer(const unsigned int updateEvery) {
  // updateEvery is how often the updateRPM will be called.
  toRPSMultiplier = 1000 / updateEvery;
}

void Tachometer::incrementCounter() {
  counter++;
}

void Tachometer::updateRPM() {
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
      // round per interval * (1000/interval) == RPS
      // counter / 2 == rounds per interval
      // RPS * 60 == RPM
      rpm = ((counter / 2) * toRPSMultiplier) * 60;
      counter = 0;
    }
}

