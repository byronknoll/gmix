#include "stationary-map.h"

#include <algorithm>

StationaryMapMemory::StationaryMapMemory(unsigned int n_contexts,
                                         std::string desc)
    : description(desc), table(n_contexts, 32768), num_contexts(n_contexts) {}

void StationaryMapMemory::WriteToDisk(std::ofstream* s) {
  Serialize(s, num_contexts);
  SerializeArray(s, table);
}

void StationaryMapMemory::ReadFromDisk(std::ifstream* s) {
  Serialize(s, num_contexts);
  SerializeArray(s, table);
}

void StationaryMapMemory::Copy(const MemoryInterface* m) {
  const StationaryMapMemory* orig =
      static_cast<const StationaryMapMemory*>(m);
  description = orig->description;
  num_contexts = orig->num_contexts;
  table = orig->table;
}

StationaryMap::StationaryMap(ShortTermMemory& short_term_memory,
                             LongTermMemory& long_term_memory,
                             const unsigned int& context,
                             unsigned int num_contexts, int rate,
                             std::string description, bool enable_analysis)
    : context_(context), num_contexts_(num_contexts), rate_(rate) {
  prediction_index_ =
      short_term_memory.AddPrediction(description, enable_analysis, this);
  memory_index_ = long_term_memory.model_memory.size();
  long_term_memory.model_memory.push_back(
      std::make_unique<StationaryMapMemory>(num_contexts_, description));
}

StationaryMapMemory* StationaryMap::GetMemory(LongTermMemory& long_term_memory) {
  return static_cast<StationaryMapMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

const StationaryMapMemory* StationaryMap::GetMemory(
    const LongTermMemory& long_term_memory) const {
  return static_cast<const StationaryMapMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

void StationaryMap::Predict(ShortTermMemory& short_term_memory,
                            const LongTermMemory& long_term_memory) {
  unsigned int full_ctx =
      (context_ << 8) | (short_term_memory.bit_context & 0xff);
  unsigned int idx = full_ctx % num_contexts_;
  const auto& mem = *GetMemory(long_term_memory);
  uint16_t val = mem.table[idx];
  float p = (static_cast<float>(val) + 0.5f) / 65536.0f;
  short_term_memory.SetPrediction(p, prediction_index_);
  last_index_ = idx;
}

void StationaryMap::Learn(const ShortTermMemory& short_term_memory,
                          LongTermMemory& long_term_memory) {
  auto& mem = *GetMemory(long_term_memory);
  int y = short_term_memory.new_bit;
  int target = y ? 65535 : 0;
  int current = mem.table[last_index_];
  mem.table[last_index_] = current + ((target - current) >> rate_);
}

void StationaryMap::WriteToDisk(std::ofstream* s) {
  Serialize(s, last_index_);
}

void StationaryMap::ReadFromDisk(std::ifstream* s) {
  Serialize(s, last_index_);
}

void StationaryMap::Copy(const MemoryInterface* m) {
  const StationaryMap* orig = static_cast<const StationaryMap*>(m);
  last_index_ = orig->last_index_;
}

unsigned long long StationaryMap::GetMemoryUsage(
    const ShortTermMemory& short_term_memory,
    const LongTermMemory& long_term_memory) {
  const auto& mem = *GetMemory(long_term_memory);
  return sizeof(*this) + sizeof(StationaryMapMemory) +
         mem.table.size() * sizeof(uint16_t);
}
