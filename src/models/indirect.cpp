#include "indirect.h"

#include <stdlib.h>

Indirect::Indirect(ShortTermMemory& short_term_memory,
                   LongTermMemory& long_term_memory, float learning_rate,
                   unsigned int table_size, unsigned int& context,
                   std::string description, bool enable_analysis)
    : context_(context), learning_rate_(learning_rate) {
  prediction_index_indirect_ = short_term_memory.AddPrediction(
      description + "-indirect", enable_analysis, this);
  prediction_index_run_map_ = short_term_memory.AddPrediction(
      description + "-run_map", enable_analysis, this);
  memory_index_ = long_term_memory.model_memory.size();
  // When the table size is a multiple of 256, there will be more context
  // collisions (because the byte context index will always be a multiple of
  // 256). By adding 1 to the table size we can spread out the byte contexts to
  // create fewer collisions.
  long_term_memory.model_memory.push_back(
      std::make_unique<IndirectMemory>(table_size * 256 + 1, description));
  IndirectMemory* mem = GetMemory(long_term_memory);
  for (int i = 0; i < 256; ++i) {
    mem->nonstationary_predictions[i] = 0;
  }
  for (int i = 0; i < 256; ++i) {
    mem->run_map_predictions[i] = 0;
  }
}

IndirectMemory* Indirect::GetMemory(LongTermMemory& long_term_memory) {
  return static_cast<IndirectMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

const IndirectMemory* Indirect::GetMemory(
    const LongTermMemory& long_term_memory) const {
  return static_cast<const IndirectMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

void Indirect::Predict(ShortTermMemory& short_term_memory,
                       const LongTermMemory& long_term_memory) {
  const auto& m = *GetMemory(long_term_memory);
  unsigned int context =
      ((static_cast<uint64_t>(context_) << 8) + short_term_memory.bit_context) %
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
  auto& m = *GetMemory(long_term_memory);
  unsigned int context =
      ((static_cast<uint64_t>(context_) << 8) + short_term_memory.bit_context) %
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
  usage += 2 * GetMemory(long_term_memory)->run_map_table.size();
  return usage;
}

void IndirectMemory::WriteToDisk(std::ofstream* s) {
  std::vector<unsigned int> keys;
  for (int i = 0; i < nonstationary_table.size(); ++i) {
    if (nonstationary_table[i] != 255) {
      keys.push_back(i);
    }
  }
  unsigned int size = keys.size();
  Serialize(s, size);
  if (size < nonstationary_table.size() / 3) {
    // If the table is sparse, encode keys+values.
    for (unsigned int key : keys) {
      Serialize(s, key);
      Serialize(s, nonstationary_table[key]);
      Serialize(s, run_map_table[key]);
    }
  } else {
    // If the table is dense, encode all values.
    SerializeArray(s, nonstationary_table);
    SerializeArray(s, run_map_table);
  }

  SerializeArray(s, nonstationary_predictions);
  SerializeArray(s, run_map_predictions);
}

void IndirectMemory::ReadFromDisk(std::ifstream* s) {
  unsigned int size;
  Serialize(s, size);
  if (size < nonstationary_table.size() / 3) {
    // If the table is sparse, encode keys+values.
    for (int i = 0; i < size; ++i) {
      unsigned int key;
      Serialize(s, key);
      unsigned char state;
      Serialize(s, state);
      nonstationary_table[key] = state;
      Serialize(s, state);
      run_map_table[key] = state;
    }
  } else {
    // If the table is dense, encode all values.
    SerializeArray(s, nonstationary_table);
    SerializeArray(s, run_map_table);
  }
  SerializeArray(s, nonstationary_predictions);
  SerializeArray(s, run_map_predictions);
}

void IndirectMemory::Copy(const MemoryInterface* m) {
  const IndirectMemory* orig = static_cast<const IndirectMemory*>(m);
  description = orig->description;
  nonstationary_table = orig->nonstationary_table;
  run_map_table = orig->run_map_table;
  nonstationary_predictions = orig->nonstationary_predictions;
  run_map_predictions = orig->run_map_predictions;
}