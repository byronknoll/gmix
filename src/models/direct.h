#ifndef MODELS_DIRECT_H_
#define MODELS_DIRECT_H_

#include "../model.h"

// This simple model directly maps each context to a probability. This model only
// supports small contexts (up to 24 bits).
class Direct : public Model {
 public:
  // limit: as the context count gets closer to this limit, the learning rate
  // decreases.
  Direct(ShortTermMemory& short_term_memory, LongTermMemory& long_term_memory,
         int limit, unsigned long long& context);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory);
  void WriteToDisk(std::ofstream* s) {}
  void ReadFromDisk(std::ifstream* s) {}
  void Copy(const MemoryInterface* m) {}

 private:
  int limit_, prediction_index_, memory_index_;
  float min_learning_rate_;
  unsigned long long& context_;
};

#endif  // MODELS_DIRECT_H_