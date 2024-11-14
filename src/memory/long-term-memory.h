#ifndef LONG_TERM_MEMORY_H_
#define LONG_TERM_MEMORY_H_

#include <array>
#include <memory>
#include <unordered_map>
#include <valarray>
#include <vector>

#include "../memory-interface.h"

struct IndirectMemory {
  IndirectMemory(unsigned long long table_size)
      : nonstationary_table(table_size, 255), run_map_table(table_size, 0) {
    nonstationary_table.shrink_to_fit();
    run_map_table.shrink_to_fit();
  }
  // Map from context to nonstationary state:
  std::vector<unsigned char> nonstationary_table;
  // Map from context to run map state:
  std::vector<unsigned char> run_map_table;
  // Map from state to prediction (in logit space).
  std::array<float, 256> nonstationary_predictions;
  // Map from state to prediction (in logit space).
  std::array<float, 256> run_map_predictions;
};

struct MixerData {
  MixerData(unsigned int input_size) : steps(0), weights(input_size) {};
  unsigned long long steps;
  std::valarray<float> weights;
};

struct MixerMemory {
  std::unordered_map<unsigned int, std::unique_ptr<MixerData>> mixer_map;
};

struct MatchMemory {
  // Map from context to "history" pointers. Each pointer is five bytes.
  std::unordered_map<unsigned int, std::array<unsigned char, 5>> map;
  // Index is the match length, value is the probability (in logit space).
  // Longer match = more probability.
  std::array<float, 256> predictions;
  // The number of times this match length has been observed.
  std::array<int, 256> counts;
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

  std::vector<IndirectMemory> indirect;
  std::vector<MixerMemory> mixers;

  // LSTM weights.
  std::vector<NeuronLayerWeights> neuron_layer_weights;
  std::valarray<std::valarray<std::valarray<float>>> lstm_output_layer;

  // A complete history of every byte of input.
  std::vector<unsigned char> history;

  std::vector<MatchMemory> match_memory;
};

#endif  // LONG_TERM_MEMORY_H_
