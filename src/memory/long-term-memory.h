#ifndef LONG_TERM_MEMORY_H_
#define LONG_TERM_MEMORY_H_

#include <array>
#include <memory>
#include <unordered_map>
#include <valarray>
#include <vector>

#include "../memory-interface.h"

struct IndirectMemory {
  // Map from context to state (contexts/nonstationary.h).
  std::unordered_map<unsigned int, unsigned char> map;
};

struct MixerData {
  MixerData(unsigned long long input_size) : steps(0), weights(input_size) {};
  unsigned long long steps;
  std::valarray<float> weights;
};

struct MixerMemory {
  std::unordered_map<unsigned int, std::unique_ptr<MixerData>> mixer_map;
};

struct NeuronLayerWeights {
  NeuronLayerWeights(unsigned int input_size, unsigned int num_cells)
      : weights(std::valarray<float>(input_size), num_cells) {};
  std::valarray<std::valarray<float>> weights;
};

// LongTermMemory contains any data/information that models use for
// training/learning.
struct LongTermMemory : MemoryInterface {
 public:
  LongTermMemory() {}
  ~LongTermMemory() {}
  void WriteToDisk(std::ofstream* s);
  void ReadFromDisk(std::ifstream* s);
  void Copy(const MemoryInterface* m);

  std::vector<std::unique_ptr<IndirectMemory>> indirect;
  std::vector<std::unique_ptr<MixerMemory>> mixers;

  // LSTM weights.
  std::vector<std::unique_ptr<NeuronLayerWeights>> neuron_layer_weights;
  std::valarray<std::valarray<std::valarray<float>>> lstm_output_layer;

  // Pointer to start of PPM allocated memory.
  unsigned char* ppmd_memory;
  // Number of bytes of PPM allocated memory.
  unsigned long long ppmd_memory_size = 0;
};

#endif  // LONG_TERM_MEMORY_H_
