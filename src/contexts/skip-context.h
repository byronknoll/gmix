#ifndef CONTEXTS_SKIP_CONTEXT_H_
#define CONTEXTS_SKIP_CONTEXT_H_

#include "../model.h"

// This context is created by taking a hash over some user-specified bytes of
// the input history. The bytes can be non-consecutive (i.e. "skip").
class SkipContext : public Model {
 public:
  // bytes_to_use: which bytes of recent history to use for the context hash.
  // This can contain up to 8 elements.
  // bytes_to_use={0}: last byte
  // bytes_to_use={1}: 2nd last byte
  // bytes_to_use={0,1}: last two bytes
  SkipContext(const std::vector<int>& bytes_to_use,
              unsigned int& output_context);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory) {}
  void WriteToDisk(std::ofstream* s) {}
  void ReadFromDisk(std::ifstream* s) {}
  void Copy(const MemoryInterface* m) {}
  unsigned long long GetMemoryUsage(const ShortTermMemory& short_term_memory,
                                    const LongTermMemory& long_term_memory) {
    return bytes_to_use_.size() * 4 + 4;
  }

 private:
  unsigned int& context_;
  std::vector<int> bytes_to_use_;
};

#endif  // CONTEXTS_SKIP_CONTEXT_H_
