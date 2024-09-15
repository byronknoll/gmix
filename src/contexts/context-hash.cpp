#include "context-hash.h"

ContextHash::ContextHash(unsigned int order, unsigned int hash_size,
                         unsigned long long& output,
                         unsigned long long& max_size)
    : hash_size_(hash_size), output_(output), size_(max_size) {
  size_ = (unsigned long long)1 << (hash_size * order);
}

void ContextHash::Perceive(ShortTermMemory& short_term_memory,
                           const LongTermMemory& long_term_memory) {
  if (short_term_memory.bit_context == 1) {
    output_ =
        (output_ * (1 << hash_size_) + short_term_memory.last_byte) % size_;
  }
}
