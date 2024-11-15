#include "skip-context.h"

#include "murmur-hash.h"

SkipContext::SkipContext(const std::vector<int>& bytes_to_use,
                         unsigned int& output_context)
    : context_(output_context), bytes_to_use_(bytes_to_use) {}

void SkipContext::Predict(ShortTermMemory& short_term_memory,
                          const LongTermMemory& long_term_memory) {
  if (short_term_memory.recent_bits == 1) {  // byte boundary.
    unsigned long long context = 0;
    for (int i = 0; i < bytes_to_use_.size(); ++i) {
      context =
          (context << 8) + short_term_memory.GetRecentByte(bytes_to_use_[i]);
    }
    MurmurHash3_x86_32(&context, 8, 0XDEADBEEF, &context_);
  }
}