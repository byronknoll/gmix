#ifndef SHORT_TERM_MEMORY_H_
#define SHORT_TERM_MEMORY_H_

#include "memory-interface.h"

#include <vector>

struct ContextHashOutput {
  unsigned long long context = 0, max_size;
};

struct ShortTermMemory : MemoryInterface {
 public:
  ShortTermMemory() {}
  ~ShortTermMemory() {}
  void WriteToDisk() {}
  void ReadFromDisk() {}

  std::vector<float> predictions;

  // Newly perceived bit.
  int new_bit = 0;

  // Ranges in value from 1 to 255.
  // 1: 0 bits seen
  // 2-3: 1 bit seen: (2=zero, 3=one)
  // ...
  // >=128: 7 bits seen
  int bit_context = 1;

  int last_byte = 0;

  ContextHashOutput hash_1_8;
};

#endif  // SHORT_TERM_MEMORY_H_
