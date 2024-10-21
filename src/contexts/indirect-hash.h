#ifndef CONTEXTS_INDIRECT_HASH_H_
#define CONTEXTS_INDIRECT_HASH_H_

#include "../model.h"

// This maps an "outer" context to an "inner" context. The outer context is
// based on recently seen bytes. The inner context is based on the bytes seen
// *after* that outer context match.
class IndirectHash : public Model {
 public:
  // outer_order: number of bytes to use for the outer context.
  // outer_hash_size: number of bits to use for the outer context hash.
  // inner_order: number of bytes to use for the inner context.
  // inner_hash_size: number of bits to use for the inner context hash.
  // output_context: reference for where to store the matched inner context.
  IndirectHash(int outer_order, int outer_hash_size, int inner_order,
               int inner_hash_size, unsigned int& output_context);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory) {}
  void WriteToDisk(std::ofstream* s);
  void ReadFromDisk(std::ifstream* s);
  void Copy(const MemoryInterface* m);
  unsigned long long GetMemoryUsage(const ShortTermMemory& short_term_memory,
                                    const LongTermMemory& long_term_memory);

 private:
  // Map from outer context to inner context.
  std::unordered_map<unsigned int, unsigned long long> map_;
  unsigned long long outer_context_ = 0;
  // These are used to truncate the outer/inner contexts to the correct number
  // of bits (based on the context order).
  unsigned long long outer_mod_, inner_mod_;
  unsigned int outer_hash_ = 0;
  // These are used to truncate the context hash to the correct number of bits.
  unsigned int outer_hash_mod_, inner_hash_mod_;
  unsigned int& context_;
};

#endif  // CONTEXTS_INDIRECT_HASH_H_
