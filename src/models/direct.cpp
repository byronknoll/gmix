#include "direct.h"

Direct::Direct(ShortTermMemory& short_term_memory,
               LongTermMemory& long_term_memory, int limit, float delta,
               unsigned long long& byte_context, unsigned long long size)
    : limit_(limit),
      delta_(delta),
      divisor_(1.0 / (limit + delta)),
      byte_context_(byte_context) {
  prediction_index_ = short_term_memory.predictions.size();
  short_term_memory.predictions.push_back(0.5);
  long_term_memory.direct.predictions.resize(size, std::array<float, 256>());
  long_term_memory.direct.counts.resize(size, std::array<unsigned char, 256>());

  for (int i = 0; i < size; ++i) {
    long_term_memory.direct.predictions[i].fill(0.5);
    long_term_memory.direct.counts[i].fill(0);
  }
}

void Direct::Predict(ShortTermMemory& short_term_memory,
                     const LongTermMemory& long_term_memory) {
  short_term_memory.predictions[prediction_index_] =
      long_term_memory.direct
          .predictions[byte_context_][short_term_memory.bit_context];
}

void Direct::Learn(const ShortTermMemory& short_term_memory,
                   LongTermMemory& long_term_memory) {
  float divisor = divisor_;
  if (long_term_memory.direct
          .counts[byte_context_][short_term_memory.bit_context] < limit_) {
    ++long_term_memory.direct
          .counts[byte_context_][short_term_memory.bit_context];
    divisor = 1.0 / (long_term_memory.direct
                         .counts[byte_context_][short_term_memory.bit_context] +
                     delta_);
  }
  long_term_memory.direct
      .predictions[byte_context_][short_term_memory.bit_context] +=
      (short_term_memory.new_bit -
       long_term_memory.direct
           .predictions[byte_context_][short_term_memory.bit_context]) *
      divisor;
}