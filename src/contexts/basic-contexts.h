#ifndef CONTEXTS_BASIC_CONTEXTS_H_
#define CONTEXTS_BASIC_CONTEXTS_H_

#include "../model.h"

// This is used to create a set of simple contexts.
class BasicContexts : public Model {
 public:
  BasicContexts() {}
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory) {}
  void WriteToDisk(std::ofstream* os);
  void ReadFromDisk(std::ifstream* is);
 
 private:
  bool first_prediction_ = true;
};

#endif  // CONTEXTS_BASIC_CONTEXTS_H_
