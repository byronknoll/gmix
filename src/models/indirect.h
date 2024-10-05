#ifndef MODELS_INDIRECT_H
#define MODELS_INDIRECT_H

#include <array>
#include <vector>

#include "../model.h"

// This model maps contexts to a one byte "state" (see
// contexts/nonstationary.h). The state is then mapped to a 0-1 probability.
// This model only supports contexts up to 24 bits.
class Indirect : public Model {
 public:
  Indirect(ShortTermMemory& short_term_memory, LongTermMemory& long_term_memory,
           float learning_rate, unsigned long long& context);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory);
  void WriteToDisk(std::ofstream* s);
  void ReadFromDisk(std::ifstream* s);
  void Copy(const MemoryInterface* m);

 private:
  unsigned long long& context_;
  int prediction_index_, memory_index_;
  float learning_rate_;
  std::array<float, 256> predictions_;
};

#endif  // MODELS_INDIRECT_H
