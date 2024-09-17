#include "direct.h"

Direct::Direct(ShortTermMemory& short_term_memory,
               LongTermMemory& long_term_memory, int limit, float delta,
               RecentBytesContextOutput& context, DirectMemory& memory)
    : limit_(limit),
      delta_(delta),
      divisor_(1.0 / (limit + delta)),
      context_(context),
      memory_(memory) {
  prediction_index_ = short_term_memory.predictions.size();
  short_term_memory.predictions.push_back(0.5);
  memory_.predictions.resize(context.max_size, std::array<float, 255>());
  memory_.counts.resize(context.max_size, std::array<unsigned char, 255>());

  for (int i = 0; i < context.max_size; ++i) {
    memory_.predictions[i].fill(0.5);
    memory_.counts[i].fill(0);
  }
}

void Direct::Predict(ShortTermMemory& short_term_memory,
                     const LongTermMemory& long_term_memory) {
  short_term_memory.predictions[prediction_index_] =
      memory_.predictions[context_.context][short_term_memory.bit_context];
}

void Direct::Learn(const ShortTermMemory& short_term_memory,
                   LongTermMemory& long_term_memory) {
  float divisor = divisor_;
  if (memory_.counts[context_.context][short_term_memory.bit_context] <
      limit_) {
    ++memory_.counts[context_.context][short_term_memory.bit_context];
    divisor =
        1.0 / (memory_.counts[context_.context][short_term_memory.bit_context] +
               delta_);
  }
  memory_.predictions[context_.context][short_term_memory.bit_context] +=
      (short_term_memory.new_bit -
       memory_.predictions[context_.context][short_term_memory.bit_context]) *
      divisor;
}