#include "run-map.h"

RunMap::RunMap() {
  for (int i = 0; i < 512; ++i) {
    int state = i / 2;
    if (i % 2 == 0) {
      if (state < 127)
        ++state;
      else if (state >= 128)
        state = 1;
    } else {
      if (state < 128)
        state = 128;
      else if (state < 255)
        ++state;
    }
    table_[i] = state;
  }
}

int RunMap::Next(int state, int bit) const { return table_[state * 2 + bit]; }