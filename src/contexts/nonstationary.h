#ifndef CONTEXTS_NONSTATIONARY_H
#define CONTEXTS_NONSTATIONARY_H

#include <array>

// Table of state transitions for indirect model. The state is one byte and
// represents a bit history. State range: 0-254.
class Nonstationary {
 public:
  Nonstationary();
  // Given a state and the next bit of input, this returns the new state.
  int Next(int state, int bit) const;

 private:
  std::array<std::array<unsigned char, 2>, 256> table_;
};

#endif  // CONTEXTS_NONSTATIONARY_H