#include "context-hash.h"

ContextHash::ContextHash(unsigned int order, unsigned int hash_size,
                         unsigned long long& output)
    : hash_size_(hash_size), output_(output) {
  size_ = (unsigned long long)1 << (hash_size * order);
}

void ContextHash::Perceive(ShortTermMemory& short_term_memory,
                           const LongTermMemory& long_term_memory) {
  output_ =
      (output_ * (1 << hash_size_) + short_term_memory.bit_context) % size_;
}

