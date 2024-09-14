#ifndef SHORT_TERM_MEMORY_H_
#define SHORT_TERM_MEMORY_H_

struct ContextHashOutput {
  unsigned long long hash_1_8;
};

struct ShortTermMemory {
  // Ranges in value from 1 to 255.
  // 1: 0 bits seen
  // 2-3: 1 bit seen: (2=zero, 3=one)
  // ...
  // >=128: 7 bits seen
  int bit_context;

  // Newly perceived bit.
  int new_bit;

  ContextHashOutput context_hash;
};

#endif  // SHORT_TERM_MEMORY_H_
