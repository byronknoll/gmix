#ifndef CONTEXTS_INTERVAL_CONTEXT_H_
#define CONTEXTS_INTERVAL_CONTEXT_H_

#include "../model.h"

// This context uses the provided mapping to "shrink" bytes to a smaller set of
// states.
class IntervalContext : public Model {
 public:
  // map: Should contain 256 entries, mapping each byte value to a new state.
  // For example, a map can shrink 256 possible byte values to 8 different
  // states (0-7).
  // num_bits: This is the number of bits the context will use. This should be
  // >= the number of bits used to represent a state. e.g. if the map uses 8
  // states (i.e. 3 bits), setting num_bits=6 would mean the context represents
  // two bytes (using 3 bits each).
  IntervalContext(const std::vector<int>& map, unsigned int num_bits,
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
    return 256 * 4 + 8 + 4;
  }

 private:
  unsigned int& context_;  // This is the output context.
  std::vector<int> map_;   // This maps bytes to a smaller set of states.
  // This is used to limit the context to the correct number of bits.
  unsigned long long mask_;
  // This is used to shift the context by the correct number of bits for the
  // next byte.
  int shift_;
};

#endif  // CONTEXTS_INTERVAL_CONTEXT_H_
