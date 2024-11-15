#include "indirect.h"

#include <stdlib.h>

Indirect::Indirect(ShortTermMemory& short_term_memory,
                   LongTermMemory& long_term_memory, float learning_rate,
                   unsigned long long table_size, unsigned int& context,
                   std::string description, bool enable_analysis)
    : context_(context), learning_rate_(learning_rate) {
  prediction_index_indirect_ = short_term_memory.AddPrediction(
      description + "-indirect", enable_analysis, this);
  prediction_index_run_map_ = short_term_memory.AddPrediction(
      description + "-run_map", enable_analysis, this);
  memory_index_ = long_term_memory.indirect.size();
  // When the table size is a multiple of 256, there will be more context
  // collisions (because the byte context index will always be a multiple of
  // 256). By adding 1 to the table size we can spread out the byte contexts to
  // create fewer collisions.
  long_term_memory.indirect.push_back(IndirectMemory(table_size * 256 + 1));
  for (int i = 0; i < 256; ++i) {
    long_term_memory.indirect.back().nonstationary_predictions[i] = 0;
  }
  for (int i = 0; i < 256; ++i) {
    long_term_memory.indirect.back().run_map_predictions[i] = 0;
  }
}

void Indirect::Predict(ShortTermMemory& short_term_memory,
                       const LongTermMemory& long_term_memory) {
  const auto& m = long_term_memory.indirect[memory_index_];
  unsigned int context = ((context_ << 8) + short_term_memory.bit_context) %
                         m.nonstationary_table.size();
  int nonstationary_state = m.nonstationary_table[context];
  // 255 means this context has never been seen.
  if (nonstationary_state != 255) {
    float p = m.nonstationary_predictions[nonstationary_state];
    short_term_memory.SetLogitPrediction(p, prediction_index_indirect_);
  }
  int run_map_state = m.run_map_table[context];
  // 0 means this context has never been seen.
  if (run_map_state != 0) {
    float p = m.run_map_predictions[run_map_state];
    short_term_memory.SetLogitPrediction(p, prediction_index_run_map_);
  }
}

void Indirect::Learn(const ShortTermMemory& short_term_memory,
                     LongTermMemory& long_term_memory) {
  auto& m = long_term_memory.indirect[memory_index_];
  unsigned int context = ((context_ << 8) + short_term_memory.bit_context) %
                         m.nonstationary_table.size();
  int nonstationary_state = m.nonstationary_table[context];
  if (nonstationary_state == 255) {
    // 255 is the uninitialized state, so we the reset to a valid "0" state.
    nonstationary_state = 0;
  }
  m.nonstationary_predictions[nonstationary_state] +=
      (short_term_memory.new_bit -
       Sigmoid::Logistic(m.nonstationary_predictions[nonstationary_state])) *
      learning_rate_;
  m.nonstationary_table[context] = short_term_memory.nonstationary.Next(
      nonstationary_state, short_term_memory.new_bit);
  int run_map_state = m.run_map_table[context];
  m.run_map_predictions[run_map_state] +=
      (short_term_memory.new_bit -
       Sigmoid::Logistic(m.run_map_predictions[run_map_state])) *
      learning_rate_;
  m.run_map_table[context] =
      short_term_memory.run_map.Next(run_map_state, short_term_memory.new_bit);
}

unsigned long long Indirect::GetMemoryUsage(
    const ShortTermMemory& short_term_memory,
    const LongTermMemory& long_term_memory) {
  unsigned long long usage = 12;
  usage += 256 * 4 * 2;  // predictions
  usage += 2 * long_term_memory.indirect[memory_index_].run_map_table.size();
  return usage;
}