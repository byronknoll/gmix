#include "direct.h"

Direct::Direct(ShortTermMemory& short_term_memory, int limit,
               unsigned long long& context, unsigned long long size,
               DirectMemory& memory)
    : limit_(limit),
      min_learning_rate_(1.0 / limit),
      context_(context),
      memory_(memory) {
  prediction_index_ = short_term_memory.predictions.size();
  ++short_term_memory.num_predictions;
  memory_.predictions.resize(size, std::array<float, 255>());
  memory_.counts.resize(size, std::array<unsigned char, 255>());

  for (int i = 0; i < size; ++i) {
    memory_.predictions[i].fill(0.5);
    memory_.counts[i].fill(0);
  }
}

void Direct::Predict(ShortTermMemory& short_term_memory,
                     const LongTermMemory& long_term_memory) {
  short_term_memory.predictions[prediction_index_] =
      memory_.predictions[context_][short_term_memory.bit_context];
}

void Direct::Learn(const ShortTermMemory& short_term_memory,
                   LongTermMemory& long_term_memory) {
  float learning_rate = min_learning_rate_;
  if (memory_.counts[context_][short_term_memory.bit_context] < limit_) {
    ++memory_.counts[context_][short_term_memory.bit_context];
    learning_rate =
        1.0 / memory_.counts[context_][short_term_memory.bit_context];
  }
  memory_.predictions[context_][short_term_memory.bit_context] +=
      (short_term_memory.new_bit -
       memory_.predictions[context_][short_term_memory.bit_context]) *
      learning_rate;
}