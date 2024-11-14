#ifndef CONTEXTS_RUN_MAP_H
#define CONTEXTS_RUN_MAP_H

#include <array>

// Table of state transitions for indirect model. The state keeps track of
// sequences of consecutive 0s or 1s:
// 0-127: number of consecutive "0" bits
// 128-255: number of consecutive "1" bits (after subtracting 127)
class RunMap {
 public:
  RunMap();
  // Given a state and the next bit of input, this returns the new state.
  int Next(int state, int bit) const;

 private:
  std::array<unsigned char, 512> table_;
};

#endif  // CONTEXTS_RUN_MAP_H