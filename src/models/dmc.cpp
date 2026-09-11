#include "dmc.h"

#include <algorithm>
#include <cstring>

namespace {
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

DmcMemory::DmcMemory(uint32_t max_nodes, uint32_t th_start, std::string desc)
    : description(desc), max_nodes_(max_nodes), th_start_(th_start) {
  t.resize(max_nodes);
  ResetStateGraph(th_start);
}

void DmcMemory::ResetStateGraph(uint32_t th_start) {
  top = extra = 0;
  threshold = th_start;
  threshold_fine = th_start << 11;
  sm_table.fill(1u << 31);

  for (int j = 0; j < 256; ++j) {
    for (int i = 0; i < 255; ++i) {
      if (i < 127) {
        t[top].set_nx0(top + i + 1);
        t[top].set_nx1(top + i + 2);
      } else {
        int linked_tree_root = (i - 127) * 2 * 255;
        t[top].set_nx0(linked_tree_root);
        t[top].set_nx1(linked_tree_root + 255);
      }
      t[top].c0 = t[top].c1 = (th_start < 1024 ? 2048 : 512);
      t[top].set_state(0);
      ++top;
    }
  }
}

void DmcMemory::WriteToDisk(std::ofstream* s) {
  Serialize(s, top);
  Serialize(s, threshold);
  Serialize(s, threshold_fine);
  Serialize(s, extra);
  SerializeArray(s, sm_table);
  // Write only allocated nodes
  s->write(reinterpret_cast<const char*>(t.data()), top * sizeof(DMCNode));
}

void DmcMemory::ReadFromDisk(std::ifstream* s) {
  Serialize(s, top);
  Serialize(s, threshold);
  Serialize(s, threshold_fine);
  Serialize(s, extra);
  SerializeArray(s, sm_table);
  if (top > t.size()) t.resize(top);
  s->read(reinterpret_cast<char*>(t.data()), top * sizeof(DMCNode));
}

void DmcMemory::Copy(const MemoryInterface* m) {
  const DmcMemory* orig = static_cast<const DmcMemory*>(m);
  description = orig->description;
  top = orig->top;
  threshold = orig->threshold;
  threshold_fine = orig->threshold_fine;
  extra = orig->extra;
  sm_table = orig->sm_table;
  max_nodes_ = orig->max_nodes_;
  th_start_ = orig->th_start_;
  t = orig->t;
}

DMC::DMC(ShortTermMemory& short_term_memory, LongTermMemory& long_term_memory,
         uint32_t max_nodes, uint32_t th_start, std::string description,
         bool enable_analysis, bool add_skip_connection) {
  prediction_index_counts_ = short_term_memory.AddPrediction(
      description + "-counts", enable_analysis, this);
  prediction_index_state_ = short_term_memory.AddPrediction(
      description + "-state", enable_analysis, this);
  if (add_skip_connection) {
    short_term_memory.models_with_skip_connection.push_back(
        prediction_index_counts_);
    short_term_memory.models_with_skip_connection.push_back(
        prediction_index_state_);
  }
  memory_index_ = long_term_memory.model_memory.size();
  long_term_memory.model_memory.push_back(
      std::make_unique<DmcMemory>(max_nodes, th_start, description));
}

DmcMemory* DMC::GetMemory(LongTermMemory& long_term_memory) {
  return static_cast<DmcMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

const DmcMemory* DMC::GetMemory(const LongTermMemory& long_term_memory) const {
  return static_cast<const DmcMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

void DMC::Predict(ShortTermMemory& short_term_memory,
                  const LongTermMemory& long_term_memory) {
  const auto& mem = *GetMemory(long_term_memory);
  uint32_t n0 = mem.t[curr_].c0 + 1;
  uint32_t n1 = mem.t[curr_].c1 + 1;
  float p_counts = static_cast<float>(n1) / static_cast<float>(n0 + n1);

  last_sm_cxt_ = mem.t[curr_].get_state();
  uint32_t val = mem.sm_table[last_sm_cxt_];
  float p_state = static_cast<float>(val >> 10) / static_cast<float>(1 << 22);

  short_term_memory.SetPrediction(p_counts, prediction_index_counts_);
  short_term_memory.SetPrediction(p_state, prediction_index_state_);
  short_term_memory.dmc_state_context = last_sm_cxt_;
  int dmc_bit = (p_counts >= 0.5f) ? 1 : 0;
  int ppm_bit = (short_term_memory.ppm_bit_context >> 8) & 1;
  int lstm_bit = (short_term_memory.lstm_bit_context >> 8) & 1;
  short_term_memory.bit_agreement_context =
      (dmc_bit << 10) | (ppm_bit << 9) | (lstm_bit << 8) |
      (short_term_memory.bit_context & 0xff);
}

void DMC::Learn(const ShortTermMemory& short_term_memory,
                LongTermMemory& long_term_memory) {
  auto& mem = *GetMemory(long_term_memory);
  int y = short_term_memory.new_bit;

  // Update StateMap for DMC bit history
  uint32_t p0 = mem.sm_table[last_sm_cxt_];
  int n = p0 & 1023;
  int pr = p0 >> 10;
  int target = y << 22;
  int delta = ((target - pr) >> 3) * kDTable.dt[n];
  p0 += (delta & 0xfffffc00);
  if (n < 1023) ++p0;
  mem.sm_table[last_sm_cxt_] = p0;

  // Update DMC counts and state
  uint32_t c0 = mem.t[curr_].c0;
  uint32_t c1 = mem.t[curr_].c1;
  const uint32_t count = (y == 0 ? c0 : c1);

  mem.t[curr_].c0 = ((((c0 << 6) - c0) >> 6) + ((1 - y) << 10));
  mem.t[curr_].c1 = ((((c1 << 6) - c1) >> 6) + (y << 10));
  mem.t[curr_].set_state(
      short_term_memory.nonstationary.Next(mem.t[curr_].get_state(), y));

  // Clone next state when threshold is reached
  if (count > mem.threshold) {
    const uint32_t next = (y == 0 ? mem.t[curr_].get_nx0() : mem.t[curr_].get_nx1());
    c0 = mem.t[next].c0;
    c1 = mem.t[next].c1;
    const uint32_t nn = c0 + c1;
    if (nn > count + mem.threshold) {
      if (mem.top < mem.t.size()) {
        uint32_t c0_top = (static_cast<uint64_t>(c0) * count) / nn;
        uint32_t c1_top = (static_cast<uint64_t>(c1) * count) / nn;
        c0 -= c0_top;
        c1 -= c1_top;
        mem.t[mem.top].c0 = c0_top;
        mem.t[mem.top].c1 = c1_top;
        mem.t[next].c0 = c0;
        mem.t[next].c1 = c1;
        mem.t[mem.top].set_nx0(mem.t[next].get_nx0());
        mem.t[mem.top].set_nx1(mem.t[next].get_nx1());
        mem.t[mem.top].set_state(mem.t[next].get_state());
        if (y == 0)
          mem.t[curr_].set_nx0(mem.top);
        else
          mem.t[curr_].set_nx1(mem.top);
        ++mem.top;
        if (mem.threshold < 8 * 1024)
          mem.threshold = (++mem.threshold_fine) >> 11;
      } else {
        mem.extra += (nn >> 10);
      }
    }
  }

  if (y == 0)
    curr_ = mem.t[curr_].get_nx0();
  else
    curr_ = mem.t[curr_].get_nx1();
}

void DMC::WriteToDisk(std::ofstream* s) {
  Serialize(s, curr_);
  Serialize(s, last_sm_cxt_);
}

void DMC::ReadFromDisk(std::ifstream* s) {
  Serialize(s, curr_);
  Serialize(s, last_sm_cxt_);
}

void DMC::Copy(const MemoryInterface* m) {
  const DMC* orig = static_cast<const DMC*>(m);
  curr_ = orig->curr_;
  last_sm_cxt_ = orig->last_sm_cxt_;
}

unsigned long long DMC::GetMemoryUsage(const ShortTermMemory& short_term_memory,
                                      const LongTermMemory& long_term_memory) {
  const auto& mem = *GetMemory(long_term_memory);
  return sizeof(*this) + sizeof(DmcMemory) + mem.top * sizeof(DMCNode);
}
