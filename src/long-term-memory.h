#ifndef LONG_TERM_MEMORY_H_
#define LONG_TERM_MEMORY_H_

#include <array>
#include <vector>

#include "memory-interface.h"

struct DirectMemory {
  std::vector<std::array<float, 256>> predictions;
  std::vector<std::array<unsigned char, 256>> counts;
};

struct LongTermMemory : MemoryInterface {
 public:
  LongTermMemory() {}
  ~LongTermMemory() {}
  void WriteToDisk() {}
  void ReadFromDisk() {}

  DirectMemory direct_;
};

#endif  // LONG_TERM_MEMORY_H_
