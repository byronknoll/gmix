#include "recent-bytes-context.h"

RecentBytesContext::RecentBytesContext(unsigned int order, RecentBytesContextOutput& output)
    : output_(output) {
  output_.max_size = (unsigned long long)1 << (8 * order);
}

void RecentBytesContext::Perceive(ShortTermMemory& short_term_memory,
                           const LongTermMemory& long_term_memory) {
  if (short_term_memory.recent_bits == 1) {
    output_.context =
        ((output_.context << 8) + short_term_memory.last_byte) %
        output_.max_size;
  }
}
