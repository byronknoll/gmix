#ifndef MODELS_INDIRECT_H
#define MODELS_INDIRECT_H

#include <array>
#include <vector>

#include "../model.h"

// This model maps contexts to two one byte "states" (see
// contexts/nonstationary.h and contexts/run-map.h). The state is then mapped to
// a probability. This model only supports contexts up to 24 bits.
class Indirect : public Model {
 public:
  // table_size: amount of memory to use for storing states. The context table
  // size will be 256 times larger than this value.
  // description: a short identifier for this model.
  Indirect(ShortTermMemory& short_term_memory, LongTermMemory& long_term_memory,
           float learning_rate, unsigned long long table_size,
           unsigned int& context, std::string description,
           bool enable_analysis);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory);
  void WriteToDisk(std::ofstream* s) {}
  void ReadFromDisk(std::ifstream* s) {}
  void Copy(const MemoryInterface* m) {}
  unsigned long long GetMemoryUsage(const ShortTermMemory& short_term_memory,
                                    const LongTermMemory& long_term_memory);

 private:
  unsigned int& context_;
  int prediction_index_indirect_, prediction_index_run_map_, memory_index_;
  float learning_rate_;
};

#endif  // MODELS_INDIRECT_H
