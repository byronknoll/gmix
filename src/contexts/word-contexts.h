#ifndef CONTEXTS_WORD_CONTEXTS_H_
#define CONTEXTS_WORD_CONTEXTS_H_

#include <array>
#include <cstdint>
#include <vector>

#include "../model.h"

// WordContexts discovers word/token boundaries dynamically from stream
// statistics without hardcoding specific byte meanings (such as ASCII spaces).
// It maintains word tokens and generates n-gram contexts for predictive models.
class WordContexts : public Model {
 public:
  WordContexts();

  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory) override;
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory) override {}
  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;
  unsigned long long GetMemoryUsage(
      const ShortTermMemory& short_term_memory,
      const LongTermMemory& long_term_memory) override;

 private:
  std::array<int64_t, 256> last_pos_;
  std::array<int, 256> following_variety_;
  std::array<int, 256> preceding_variety_;
  std::array<uint64_t, 1024> following_bits_;
  std::array<uint64_t, 1024> preceding_bits_;
  std::array<float, 256> separator_score_;

  int detected_separator_ = -1;
  int detected_secondary_delimiter_ = -1;
  uint32_t cur_word_ = 0;
  int cur_word_len_ = 0;
  int prev_word_len_ = 0;
  uint32_t first_byte_ = 0;
  uint32_t word_1_ = 0;
  uint32_t word_2_ = 0;
  uint32_t word_3_ = 0;
  uint32_t word_4_ = 0;
  uint32_t word_5_ = 0;
  int prev_byte_ = -1;
  int64_t byte_pos_ = 0;
};

#endif  // CONTEXTS_WORD_CONTEXTS_H_
