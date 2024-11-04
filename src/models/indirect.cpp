#include "indirect.h"

#include <stdlib.h>

Indirect::Indirect(ShortTermMemory& short_term_memory,
                   LongTermMemory& long_term_memory, float learning_rate,
                   unsigned int& context, std::string description,
                   bool enable_analysis)
    : context_(context), learning_rate_(learning_rate) {
  prediction_index_indirect_ = short_term_memory.AddPrediction(
      description + "-indirect", enable_analysis, this);
  prediction_index_run_map_ = short_term_memory.AddPrediction(
      description + "-run_map", enable_analysis, this);
  memory_index_ = long_term_memory.indirect.size();
  long_term_memory.indirect.push_back(IndirectMemory());
  for (int i = 0; i < 256; ++i) {
    long_term_memory.indirect.back().nonstationary_predictions[i] = 0.5;
  }
  for (int i = 0; i < 256; ++i) {
    long_term_memory.indirect.back().run_map_predictions[i] = 0.5;
  }
}

void Indirect::Predict(ShortTermMemory& short_term_memory,
                       const LongTermMemory& long_term_memory) {
  unsigned int context = (context_ << 8) + short_term_memory.bit_context;
  const auto& m = long_term_memory.indirect[memory_index_];
  const auto& it = m.map.find(context);
  if (it != m.map.end()) {
    float p = m.nonstationary_predictions[it->second[0]];
    short_term_memory.SetPrediction(p, prediction_index_indirect_);
    p = m.run_map_predictions[it->second[1]];
    short_term_memory.SetPrediction(p, prediction_index_run_map_);
  }
}

void Indirect::Learn(const ShortTermMemory& short_term_memory,
                     LongTermMemory& long_term_memory) {
  unsigned int context = (context_ << 8) + short_term_memory.bit_context;
  auto& m = long_term_memory.indirect[memory_index_];
  int nonstationary_state = m.map[context][0];
  m.nonstationary_predictions[nonstationary_state] +=
      (short_term_memory.new_bit -
       m.nonstationary_predictions[nonstationary_state]) *
      learning_rate_;
  m.map[context][0] = short_term_memory.nonstationary.Next(
      nonstationary_state, short_term_memory.new_bit);
  int run_map_state = m.map[context][1];
  m.run_map_predictions[run_map_state] +=
      (short_term_memory.new_bit - m.run_map_predictions[run_map_state]) *
      learning_rate_;
  m.map[context][1] =
      short_term_memory.run_map.Next(run_map_state, short_term_memory.new_bit);
}

unsigned long long Indirect::GetMemoryUsage(
    const ShortTermMemory& short_term_memory,
    const LongTermMemory& long_term_memory) {
  unsigned long long usage = 12;
  usage += 256 * 4 * 2;  // predictions
  usage += 6 * long_term_memory.indirect[memory_index_].map.size();
  return usage;
}