#ifndef MODELS_DMC_H_
#define MODELS_DMC_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "../model.h"

struct DMCNode {
  uint16_t c0 = 512;
  uint16_t c1 = 512;

 private:
  uint32_t _nx0 = 0;
  uint32_t _nx1 = 0;

 public:
  uint8_t get_state() const {
    return uint8_t(((_nx0 & 0xf) << 4) | (_nx1 & 0xf));
  }
  void set_state(uint8_t state) {
    _nx0 = (_nx0 & 0xfffffff0) | (state >> 4);
    _nx1 = (_nx1 & 0xfffffff0) | (state & 0xf);
  }
  uint32_t get_nx0() const { return _nx0 >> 4; }
  void set_nx0(uint32_t nx0) { _nx0 = (_nx0 & 0xf) | (nx0 << 4); }
  uint32_t get_nx1() const { return _nx1 >> 4; }
  void set_nx1(uint32_t nx1) { _nx1 = (_nx1 & 0xf) | (nx1 << 4); }
};

struct DmcMemory : public MemoryInterface {
  DmcMemory(uint32_t max_nodes, uint32_t th_start, std::string desc = "");
  std::string description;
  std::vector<DMCNode> t;
  std::array<uint32_t, 256> sm_table;
  uint32_t top = 0;
  uint32_t threshold = 0;
  uint32_t threshold_fine = 0;
  uint32_t extra = 0;
  uint32_t max_nodes_ = 0;
  uint32_t th_start_ = 0;

  void ResetStateGraph(uint32_t th_start);
  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;
};

// Dynamic Markov Compression (DMC) model adapted from PAQ8.
// Builds an adaptive bit-level finite-state automaton via state cloning.
// Completely general-purpose and byte-agnostic.
class DMC : public Model {
 public:
  DMC(ShortTermMemory& short_term_memory, LongTermMemory& long_term_memory,
      uint32_t max_nodes, uint32_t th_start, std::string description,
      bool enable_analysis = false, bool add_skip_connection = true);

  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory) override;
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory) override;
  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;
  unsigned long long GetMemoryUsage(
      const ShortTermMemory& short_term_memory,
      const LongTermMemory& long_term_memory) override;

  int GetPredictionIndexCounts() const { return prediction_index_counts_; }
  int GetPredictionIndexState() const { return prediction_index_state_; }

 private:
  DmcMemory* GetMemory(LongTermMemory& long_term_memory);
  const DmcMemory* GetMemory(const LongTermMemory& long_term_memory) const;

  uint32_t curr_ = 0;
  uint8_t last_sm_cxt_ = 0;
  int prediction_index_counts_ = 0;
  int prediction_index_state_ = 0;
  int memory_index_ = 0;
};

#endif  // MODELS_DMC_H_
