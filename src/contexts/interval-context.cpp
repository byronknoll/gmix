#include "interval-context.h"

IntervalContext::IntervalContext(const std::vector<int>& map,
                                 unsigned int num_bits,
                                 unsigned int& output_context)
    : context_(output_context), map_(map) {
  context_ = 0;
  int max_value = 0;  // Max state value in the map.
  for (unsigned int i = 0; i < map.size(); ++i) {
    if (map[i] > max_value) max_value = map[i];
  }
  shift_ = 1;
  while ((1 << shift_) <= max_value) ++shift_;
  mask_ = (1 << num_bits) - 1;
}

void IntervalContext::Predict(ShortTermMemory& short_term_memory,
                              const LongTermMemory& long_term_memory) {
  if (short_term_memory.recent_bits == 1) {  // byte boundary.
    context_ =
        mask_ & ((context_ << shift_) + map_[short_term_memory.last_byte]);
  }
}