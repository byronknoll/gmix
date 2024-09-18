#ifndef LONG_TERM_MEMORY_H_
#define LONG_TERM_MEMORY_H_

#include <array>
#include <vector>

#include "memory-interface.h"

struct DirectMemory {
  std::vector<std::array<float, 255>> predictions;
  std::vector<std::array<unsigned char, 255>> counts;
};

// LongTermMemory contains any data/information that models use for
// training/learning.
struct LongTermMemory : MemoryInterface {
 public:
  LongTermMemory() {}
  ~LongTermMemory() {}
  void WriteToDisk() {}
  void ReadFromDisk() {}

  DirectMemory direct_1;
  DirectMemory direct_2;
};

#endif  // LONG_TERM_MEMORY_H_
