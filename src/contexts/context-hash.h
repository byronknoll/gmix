#ifndef CONTEXTS_CONTEXT_HASH_H_
#define CONTEXTS_CONTEXT_HASH_H_

#include "../model.h"

class ContextHash : public Model {
 public:
  ContextHash(unsigned int order, unsigned int hash_size,
              unsigned long long& output);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory) {}
  void Perceive(ShortTermMemory& short_term_memory,
                const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory) {}

 private:
  unsigned int hash_size_;
  unsigned long long size_;
  unsigned long long& output_;
};

#endif  // CONTEXTS_CONTEXT_HASH_H_
