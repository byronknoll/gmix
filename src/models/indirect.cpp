#include "indirect.h"

#include <stdlib.h>

Indirect::Indirect(ShortTermMemory& short_term_memory,
                   LongTermMemory& long_term_memory, float learning_rate,
                   unsigned int& context)
    : context_(context), learning_rate_(learning_rate) {
  prediction_index_ = short_term_memory.num_predictions++;
  memory_index_ = long_term_memory.indirect.size();
  long_term_memory.indirect.push_back(IndirectMemory());
  for (int i = 0; i < 256; ++i) {
    long_term_memory.indirect.back().predictions[i] = 0.5;
  }
}

void Indirect::Predict(ShortTermMemory& short_term_memory,
                       const LongTermMemory& long_term_memory) {
  unsigned int context = (context_ << 8) + short_term_memory.bit_context;
  float p = 0.5;
  const auto& m = long_term_memory.indirect[memory_index_];
  const auto& it = m.map.find(context);
  if (it != m.map.end()) {
    p = m.predictions[it->second];
  }
  short_term_memory.SetPrediction(p, prediction_index_);
}

void Indirect::Learn(const ShortTermMemory& short_term_memory,
                     LongTermMemory& long_term_memory) {
  unsigned int context = (context_ << 8) + short_term_memory.bit_context;
  auto& m = long_term_memory.indirect[memory_index_];
  int state = m.map[context];
  m.predictions[state] +=
      (short_term_memory.new_bit - m.predictions[state]) * learning_rate_;
  m.map[context] =
      short_term_memory.nonstationary.Next(state, short_term_memory.new_bit);
}