#ifndef SHORT_TERM_MEMORY_H_
#define SHORT_TERM_MEMORY_H_

#include <vector>

#include "memory-interface.h"

struct RecentBytesContextOutput {
  // Hash of the most recent bytes. Values range from 0 to (max_size - 1).
  unsigned long long context = 0;
  unsigned long long max_size;
};

// ShortTermMemory contains "state" models need in order to make predictions,
// but does not contain any data used for training/learning. Models can also
// store state within member variables of their class, so the primary purpose of
// this struct is as a way to share inputs/outputs between models.
struct ShortTermMemory : MemoryInterface {
 public:
  ShortTermMemory() {}
  ~ShortTermMemory() {}
  void WriteToDisk() {}
  void ReadFromDisk() {}

  // Predictions for the next bit of data. Each prediction should be a
  // probability between 0 to 1.
  std::vector<float> predictions;

  // The most recently perceived bit.
  int new_bit = 0;

  // Recently perceived bits of the current byte. The leftmost "1" bit
  // indicates how many bits have been seen. Ranges in value from 1 to 255.
  // 1: 0 bits seen
  // 2-3: 1 bit seen: (2=zero, 3=one)
  // ...
  // 128-255: 7 bits seen
  // This gets updated *after* the "Perceive" and "Learn" calls.
  int recent_bits = 1;

  // This is equal to "recent_bits - 1", so has a range from 0 to 254.
  int bit_context = 0;

  // The previous byte. This gets updated after eight bits have been perceived
  // (i.e. recent_bits becomes "1").
  int last_byte = 0;

  RecentBytesContextOutput recent1;
  RecentBytesContextOutput recent2;
};

#endif  // SHORT_TERM_MEMORY_H_
