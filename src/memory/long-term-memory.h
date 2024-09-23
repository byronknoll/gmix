#ifndef LONG_TERM_MEMORY_H_
#define LONG_TERM_MEMORY_H_

#include <array>
#include <memory>
#include <unordered_map>
#include <valarray>
#include <vector>

#include "../memory-interface.h"

struct DirectMemory {
  // Predictions in 0-1 range. The 255 values are for "bit_context", the outer
  // index is for byte context.
  std::vector<std::array<float, 255>> predictions;
  // The number of times this context has been seen.
  std::vector<std::array<unsigned char, 255>> counts;
};

struct MixerData {
  MixerData(unsigned long long input_size) : steps(0), weights(input_size) {};
  unsigned long long steps;
  std::valarray<float> weights;
};

struct MixerMemory {
  std::unordered_map<unsigned int, std::unique_ptr<MixerData>> mixer_map;
};

// LongTermMemory contains any data/information that models use for
// training/learning.
struct LongTermMemory : MemoryInterface {
 public:
  LongTermMemory() {}
  ~LongTermMemory() {}
  void WriteToDisk(std::ofstream* os);
  void ReadFromDisk(std::ifstream* is);

  std::vector<std::unique_ptr<DirectMemory>> direct;
  std::vector<std::unique_ptr<MixerMemory>> mixers;

  std::valarray<std::valarray<std::valarray<float>>> lstm_output_layer;
};

#endif  // LONG_TERM_MEMORY_H_
