#include "indirect.h"

#include <stdlib.h>

Indirect::Indirect(ShortTermMemory& short_term_memory,
                   LongTermMemory& long_term_memory, float learning_rate,
                   unsigned long long& context)
    : context_(context), learning_rate_(learning_rate) {
  prediction_index_ = short_term_memory.num_predictions++;
  memory_index_ = long_term_memory.indirect.size();
  long_term_memory.indirect.push_back(
      std::unique_ptr<IndirectMemory>(new IndirectMemory()));
  for (int i = 0; i < 256; ++i) {
    predictions_[i] = 0.5;
  }
}

void Indirect::Predict(ShortTermMemory& short_term_memory,
                       const LongTermMemory& long_term_memory) {
  unsigned int context = (context_ << 8) + short_term_memory.bit_context;
  float p = 0.5;
  const auto& m = long_term_memory.indirect[memory_index_]->map;
  const auto& it = m.find(context);
  if (it != m.end()) {
    p = predictions_[it->second];
  }
  short_term_memory.SetPrediction(p, prediction_index_);
}

void Indirect::Learn(const ShortTermMemory& short_term_memory,
                     LongTermMemory& long_term_memory) {
  unsigned int context = (context_ << 8) + short_term_memory.bit_context;
  auto& m = long_term_memory.indirect[memory_index_]->map;
  int state = m[context];
  predictions_[state] +=
      (short_term_memory.new_bit - predictions_[state]) * learning_rate_;
  m[context] =
      short_term_memory.nonstationary.Next(state, short_term_memory.new_bit);
}

void Indirect::WriteToDisk(std::ofstream* s) {
  SerializeArray(s, predictions_);
}

void Indirect::ReadFromDisk(std::ifstream* s) {
  SerializeArray(s, predictions_);
}

void Indirect::Copy(const MemoryInterface* m) {
  const Indirect* orig = static_cast<const Indirect*>(m);
  predictions_ = orig->predictions_;
}