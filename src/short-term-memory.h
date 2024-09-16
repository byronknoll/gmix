#ifndef SHORT_TERM_MEMORY_H_
#define SHORT_TERM_MEMORY_H_

#include <vector>

#include "memory-interface.h"

struct ContextHashOutput {
  unsigned long long context = 0, max_size;
};

// ShortTermMemory contains all the "state" models need in order to make
// predictions, but does not contain any data used for training/learning.
struct ShortTermMemory : MemoryInterface {
 public:
  ShortTermMemory() {}
  ~ShortTermMemory() {}
  void WriteToDisk() {}
  void ReadFromDisk() {}

  std::vector<float> predictions;

  // The most recently perceived bit.
  int new_bit = 0;

  // Recently perceived bits of the current byte. The leftmost "1" bit
  // indicates how many bits have been seen. Ranges in value from 1 to 255.
  // 1: 0 bits seen
  // 2-3: 1 bit seen: (2=zero, 3=one)
  // ...
  // 128-255: 7 bits seen
  // This gets updated *after* the "Perceive" and "Learn" calls.
  int recent_bits = 1;

  // This is equal to "recent_bits - 1", so has a range from 0 to 254.
  int bit_context = 0;

  // The previous byte. This gets updated after eight bits have been perceived
  // (i.e. recent_bits becomes "1").
  int last_byte = 0;

  ContextHashOutput hash_1_8;
  ContextHashOutput hash_2_8;
};

#endif  // SHORT_TERM_MEMORY_H_
