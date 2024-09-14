#ifndef MODEL_H_
#define MODEL_H_

#include "long-term-memory/long-term-memory.h"
#include "short-term-memory.h"

class Model {
 public:
  Model() {}
  virtual ~Model() {}
  virtual void Predict(ShortTermMemory& short_term_memory,
                       const LongTermMemory& long_term_memory) = 0;
  virtual void Perceive(ShortTermMemory& short_term_memory,
                        const LongTermMemory& long_term_memory) = 0;
  virtual void Learn(const ShortTermMemory& short_term_memory,
                     LongTermMemory& long_term_memory) = 0;
};

#endif  // MODEL_H_
