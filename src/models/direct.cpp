#include "direct.h"

Direct::Direct(ShortTermMemory& short_term_memory,
               LongTermMemory& long_term_memory, int limit,
               unsigned long long& context, unsigned long long size)
    : limit_(limit), min_learning_rate_(1.0 / limit), context_(context) {
  prediction_index_ = short_term_memory.predictions.size();

  ++short_term_memory.num_predictions;

  memory_index_ = long_term_memory.direct.size();
  long_term_memory.direct.push_back(
      std::unique_ptr<DirectMemory>(new DirectMemory()));
  long_term_memory.direct.back()->predictions.resize(size,
                                                     std::array<float, 255>());
  long_term_memory.direct.back()->counts.resize(
      size, std::array<unsigned char, 255>());
  for (int i = 0; i < size; ++i) {
    long_term_memory.direct.back()->predictions[i].fill(0.5);
    long_term_memory.direct.back()->counts[i].fill(0);
  }
}

void Direct::Predict(ShortTermMemory& short_term_memory,
                     const LongTermMemory& long_term_memory) {
  short_term_memory.SetPrediction(
      long_term_memory.direct[memory_index_]
          ->predictions[context_][short_term_memory.bit_context],
      prediction_index_);
}

void Direct::Learn(const ShortTermMemory& short_term_memory,
                   LongTermMemory& long_term_memory) {
  float learning_rate = min_learning_rate_;
  if (long_term_memory.direct[memory_index_]
          ->counts[context_][short_term_memory.bit_context] < limit_) {
    ++long_term_memory.direct[memory_index_]
          ->counts[context_][short_term_memory.bit_context];
    learning_rate = 1.0 / long_term_memory.direct[memory_index_]
                              ->counts[context_][short_term_memory.bit_context];
  }
  long_term_memory.direct[memory_index_]
      ->predictions[context_][short_term_memory.bit_context] +=
      (short_term_memory.new_bit -
       long_term_memory.direct[memory_index_]
           ->predictions[context_][short_term_memory.bit_context]) *
      learning_rate;
}