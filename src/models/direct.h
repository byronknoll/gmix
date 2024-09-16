#ifndef MODELS_DIRECT_H_
#define MODELS_DIRECT_H_

#include "../model.h"

class Direct : public Model {
 public:
  Direct(ShortTermMemory& short_term_memory, LongTermMemory& long_term_memory,
         int limit, float delta, ContextHashOutput& context, DirectMemory& memory);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Perceive(ShortTermMemory& short_term_memory,
                const LongTermMemory& long_term_memory) {}
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory);

 private:
  int limit_, prediction_index_;
  float delta_, divisor_;
  ContextHashOutput& context_;
  DirectMemory& memory_;
};

#endif  // MODELS_DIRECT_H_