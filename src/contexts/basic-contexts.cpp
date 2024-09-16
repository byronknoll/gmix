#include "basic-contexts.h"

void BasicContexts::Predict(ShortTermMemory& short_term_memory,
                            const LongTermMemory& long_term_memory) {
  short_term_memory.recent_bits +=
      short_term_memory.recent_bits + short_term_memory.new_bit;
  if (short_term_memory.recent_bits > 256) {
    short_term_memory.last_byte = short_term_memory.recent_bits - 256;
    short_term_memory.recent_bits = 1;
  }
  short_term_memory.bit_context = short_term_memory.recent_bits - 1;
}
