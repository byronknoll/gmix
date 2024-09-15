#ifndef CONTEXTS_BASIC_CONTEXTS_H_
#define CONTEXTS_BASIC_CONTEXTS_H_

#include "../model.h"

class BasicContexts : public Model {
 public:
  BasicContexts() {}
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Perceive(ShortTermMemory& short_term_memory,
                const LongTermMemory& long_term_memory) {}
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory) {}
};

#endif  // CONTEXTS_BASIC_CONTEXTS_H_
