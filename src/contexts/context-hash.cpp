#include "context-hash.h"

ContextHash::ContextHash(unsigned int order, unsigned int hash_size,
                         ContextHashOutput& output)
    : hash_size_(hash_size), output_(output) {
  output_.max_size = (unsigned long long)1 << (hash_size * order);
}

void ContextHash::Perceive(ShortTermMemory& short_term_memory,
                           const LongTermMemory& long_term_memory) {
  if (short_term_memory.recent_bits == 1) {
    output_.context =
        (output_.context * (1 << hash_size_) + short_term_memory.last_byte) %
        output_.max_size;
  }
}
