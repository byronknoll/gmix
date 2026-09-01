#include "state-map.h"

#include <algorithm>

namespace {
// Precomputed inverse-count learning rate factors: dt[n] = 4096 / (n + 2)
struct DTable {
  uint32_t dt[1024];
  constexpr DTable() : dt{} {
    for (int i = 0; i < 1024; ++i) {
      dt[i] = 4096 / (i + 2);
    }
  }
};
constexpr DTable kDTable;
}  // namespace

StateMapMemory::StateMapMemory(unsigned int n_contexts, std::string desc)
    : description(desc),
      table(n_contexts, (1u << 31)),
      num_contexts(n_contexts) {}

void StateMapMemory::WriteToDisk(std::ofstream* s) {
  Serialize(s, num_contexts);
  SerializeArray(s, table);
}

void StateMapMemory::ReadFromDisk(std::ifstream* s) {
  Serialize(s, num_contexts);
  SerializeArray(s, table);
}

void StateMapMemory::Copy(const MemoryInterface* m) {
  const StateMapMemory* orig = static_cast<const StateMapMemory*>(m);
  description = orig->description;
  num_contexts = orig->num_contexts;
  table = orig->table;
}

StateMap::StateMap(ShortTermMemory& short_term_memory,
                   LongTermMemory& long_term_memory,
                   const unsigned int& context, unsigned int num_contexts,
                   int limit, std::string description, bool enable_analysis)
    : context_(context), num_contexts_(num_contexts), limit_(limit) {
  prediction_index_ =
      short_term_memory.AddPrediction(description, enable_analysis, this);
  memory_index_ = long_term_memory.model_memory.size();
  long_term_memory.model_memory.push_back(
      std::make_unique<StateMapMemory>(num_contexts_, description));
}

StateMapMemory* StateMap::GetMemory(LongTermMemory& long_term_memory) {
  return static_cast<StateMapMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

const StateMapMemory* StateMap::GetMemory(
    const LongTermMemory& long_term_memory) const {
  return static_cast<const StateMapMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

void StateMap::Predict(ShortTermMemory& short_term_memory,
                       const LongTermMemory& long_term_memory) {
  unsigned int full_ctx =
      (context_ << 8) | (short_term_memory.bit_context & 0xff);
  unsigned int idx = full_ctx % num_contexts_;
  const auto& mem = *GetMemory(long_term_memory);
  uint32_t val = mem.table[idx];
  float p = static_cast<float>(val >> 10) / static_cast<float>(1 << 22);
  short_term_memory.SetPrediction(p, prediction_index_);
  last_index_ = idx;
}

void StateMap::Learn(const ShortTermMemory& short_term_memory,
                     LongTermMemory& long_term_memory) {
  auto& mem = *GetMemory(long_term_memory);
  int y = short_term_memory.new_bit;
  uint32_t p0 = mem.table[last_index_];
  int n = p0 & 1023;
  int pr = p0 >> 10;
  int target = y << 22;
  int delta = ((target - pr) >> 3) * kDTable.dt[n];
  p0 += (delta & 0xfffffc00);
  if (n < limit_) ++p0;
  mem.table[last_index_] = p0;
}

void StateMap::WriteToDisk(std::ofstream* s) {
  Serialize(s, last_index_);
}

void StateMap::ReadFromDisk(std::ifstream* s) {
  Serialize(s, last_index_);
}

void StateMap::Copy(const MemoryInterface* m) {
  const StateMap* orig = static_cast<const StateMap*>(m);
  last_index_ = orig->last_index_;
}

unsigned long long StateMap::GetMemoryUsage(
    const ShortTermMemory& short_term_memory,
    const LongTermMemory& long_term_memory) {
  const auto& mem = *GetMemory(long_term_memory);
  return sizeof(*this) + sizeof(StateMapMemory) +
         mem.table.size() * sizeof(uint32_t);
}
