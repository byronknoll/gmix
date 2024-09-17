#ifndef CONTEXTS_RECENT_BYTES_CONTEXT_H_
#define CONTEXTS_RECENT_BYTES_CONTEXT_H_

#include "../model.h"

class RecentBytesContext : public Model {
 public:
  // order: number of previous bytes used for the context. Range >=0. When set
  // to zero, the context is always "0". The range of the output context is 0 to
  // 2^(8*order).
  RecentBytesContext(unsigned int order, RecentBytesContextOutput& output);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory) {}
  void Perceive(ShortTermMemory& short_term_memory,
                const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory) {}
  void WriteToDisk() {}
  void ReadFromDisk() {}

 private:
  RecentBytesContextOutput& output_;
};

#endif  // CONTEXTS_RECENT_BYTES_CONTEXT_H_
