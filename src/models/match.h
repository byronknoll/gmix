#ifndef MODELS_MATCH_H
#define MODELS_MATCH_H

#include <array>
#include <string>
#include <vector>

#include "../model.h"

struct MatchMemory : public MemoryInterface {
  MatchMemory(unsigned int size, std::string description = "")
      : description(description), table(size, {0, 0, 0, 0, 0}) {
    table.shrink_to_fit();
  };
  std::string description;
  // Map from context to "history" pointers. Each pointer is five bytes.
  std::vector<std::array<unsigned char, 5>> table;
  // Index is the match length, value is the probability (in logit space).
  // Longer match = more probability.
  std::array<float, 256> predictions;
  // The number of times this match length has been observed.
  std::array<int, 256> counts;

  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;
};

// This model maps contexts to a pointer in the input history. Predictions are
// made based on the subsequent bits from that history pointer. The probability
// is based on the length of the match (longer match = more probability).
// This model supports contexts up to 32 bits.
class Match : public Model {
 public:
  // table_size: the size of the table used for storing context matches.
  // limit: as the match count gets closer to this limit, the learning rate
  // decreases.
  // description: a short identifier for this model.
  Match(ShortTermMemory& short_term_memory, LongTermMemory& long_term_memory,
        unsigned int table_size, const unsigned int& byte_context,
        int limit, std::string description, bool enable_analysis);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory) override;
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory) override;
  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;
  unsigned long long GetMemoryUsage(const ShortTermMemory& short_term_memory,
                                    const LongTermMemory& long_term_memory) override;

 private:
  MatchMemory* GetMemory(LongTermMemory& long_term_memory);
  const MatchMemory* GetMemory(const LongTermMemory& long_term_memory) const;

  const unsigned int& byte_context_;
  // Position of current match in input history.
  unsigned long long cur_match_;
  // Current matched byte (from input history).
  unsigned char cur_byte_;
  // The binary "1" points to the current matched bit position.
  unsigned char bit_pos_;
  // This represents the number of consecutive bit matches (0-255).
  unsigned char match_length_;
  int limit_, prediction_index_, memory_index_;
  float learning_rate_;
};

#endif  // MODELS_MATCH_H