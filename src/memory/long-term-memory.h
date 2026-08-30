#ifndef LONG_TERM_MEMORY_H_
#define LONG_TERM_MEMORY_H_

#include <memory>
#include <vector>

#include "../memory-interface.h"

// LongTermMemory contains any data/information that models use for
// training/learning.
struct LongTermMemory : public MemoryInterface {
 public:
  LongTermMemory() {}
  ~LongTermMemory() {}
  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;

  // A history of input bytes (with some deduplication to save memory).
  std::vector<unsigned char> history;

  // Long-term memory that is fully owned/serialized by individual models.
  // Each model registers its own MemoryInterface instance during construction.
  std::vector<std::unique_ptr<MemoryInterface>> model_memory;
};

#endif  // LONG_TERM_MEMORY_H_
