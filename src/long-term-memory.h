#ifndef LONG_TERM_MEMORY_H_
#define LONG_TERM_MEMORY_H_

#include <array>
#include <memory>
#include <unordered_map>
#include <valarray>
#include <vector>

#include "memory-interface.h"

struct DirectMemory {
  std::vector<std::array<float, 255>> predictions;
  std::vector<std::array<unsigned char, 255>> counts;
};

struct MixerData {
  MixerData(unsigned long long input_size)
      : steps(0), weights(input_size) {};
  unsigned long long steps;
  std::valarray<float> weights;
};

// LongTermMemory contains any data/information that models use for
// training/learning.
struct LongTermMemory : MemoryInterface {
 public:
  LongTermMemory() {}
  ~LongTermMemory() {}
  void WriteToDisk() {}
  void ReadFromDisk() {}

  DirectMemory direct_0;
  DirectMemory direct_1;
  DirectMemory direct_2;

  std::unordered_map<unsigned int, std::unique_ptr<MixerData>> mixer_map;
};

#endif  // LONG_TERM_MEMORY_H_
