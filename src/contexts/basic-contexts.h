#ifndef CONTEXTS_BASIC_CONTEXTS_H_
#define CONTEXTS_BASIC_CONTEXTS_H_

#include "../model.h"

#include <array>

// This is used to create a set of simple contexts.
class BasicContexts : public Model {
 public:
  BasicContexts() {}
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory);
  void WriteToDisk(std::ofstream* s);
  void ReadFromDisk(std::ifstream* s);
  void Copy(const MemoryInterface* m);
  unsigned long long GetMemoryUsage(const ShortTermMemory& short_term_memory,
                                    const LongTermMemory& long_term_memory) {
    return sizeof(*this);
  }

 private:
  void ByteUpdate(ShortTermMemory& short_term_memory,
                  const LongTermMemory& long_term_memory);
  bool first_prediction_ = true;
  std::array<int, 256> cpos1_{};
  std::array<int, 256> cpos2_{};
  std::array<int, 256> cpos3_{};
  int candidate_stride_ = 0;
  int candidate_count_ = 0;
  int detected_stride_ = 0;
  int byte_pos_ = 0;
};

#endif  // CONTEXTS_BASIC_CONTEXTS_H_
