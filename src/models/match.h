#ifndef MODELS_MATCH_H
#define MODELS_MATCH_H

#include <array>
#include <vector>

#include "../model.h"

// This model maps contexts to a pointer in the input history. Predictions are
// made based on the subsequent bits from that history pointer. The probability
// is based on the length of the match (longer match = more probability).
// This model supports contexts up to 32 bits.
class Match : public Model {
 public:
  // limit: as the match count gets closer to this limit, the learning rate
  // decreases.
  // description: a short identifier for this model.
  Match(ShortTermMemory& short_term_memory, LongTermMemory& long_term_memory,
        const unsigned int& byte_context, int limit, std::string description);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory);
  void WriteToDisk(std::ofstream* s);
  void ReadFromDisk(std::ifstream* s);
  void Copy(const MemoryInterface* m);

 private:
  const unsigned int& byte_context_;
  // Pointer to current match in input history.
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