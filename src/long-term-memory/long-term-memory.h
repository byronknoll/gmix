#ifndef LONG_TERM_MEMORY_H_
#define LONG_TERM_MEMORY_H_

#include "direct-memory.h"
#include "long-term-memory-interface.h"

class LongTermMemory : LongTermMemoryInterface {
 public:
  LongTermMemory() {}
  ~LongTermMemory() {}
  void WriteToDisk();
  void ReadFromDisk();

  DirectMemory direct_;
};

#endif  // LONG_TERM_MEMORY_H_
