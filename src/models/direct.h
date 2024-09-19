#ifndef MODELS_DIRECT_H_
#define MODELS_DIRECT_H_

#include "../model.h"

// This simple model directly maps each context to a probability.
class Direct : public Model {
 public:
  // limit: as the context count gets closer to this limit, the learning rate
  // decreases.
  // size: context should be in the range: 0 to (size-1).
  Direct(ShortTermMemory& short_term_memory, int limit,
         unsigned long long& context, unsigned long long size,
         DirectMemory& memory);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory);
  void WriteToDisk(std::ofstream* os) {}
  void ReadFromDisk(std::ifstream* is) {}

 private:
  int limit_, prediction_index_;
  float min_learning_rate_;
  unsigned long long& context_;
  DirectMemory& memory_;
};

#endif  // MODELS_DIRECT_H_