#ifndef LONG_TERM_MEMORY_DIRECT_MEMORY_H_
#define LONG_TERM_MEMORY_DIRECT_MEMORY_H_

#include <array>
#include <vector>

#include "long-term-memory-interface.h"

class DirectMemory : public LongTermMemoryInterface {
 public:
  DirectMemory() {}
  ~DirectMemory() {}
  void WriteToDisk();
  void ReadFromDisk();

 private:
  std::vector<std::array<float, 256>> predictions_;
  std::vector<std::array<unsigned char, 256>> counts_;
};

#endif  // LONG_TERM_MEMORY_DIRECT_MEMORY_H_
