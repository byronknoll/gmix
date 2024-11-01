#ifndef CONTEXTS_RUN_MAP_H
#define CONTEXTS_RUN_MAP_H

#include <array>

// Table of state transitions for indirect model. The state is one byte and
// represents a bit history.
class RunMap {
 public:
  RunMap();
  // Given a state and the next bit of input, this returns the new state.
  int Next(int state, int bit) const;

 private:
  std::array<unsigned char, 512> table_;
};

#endif  // CONTEXTS_RUN_MAP_H