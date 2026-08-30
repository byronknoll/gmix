#ifndef MODELS_INDIRECT_H
#define MODELS_INDIRECT_H

#include <array>
#include <string>
#include <vector>

#include "../model.h"

struct IndirectMemory : public MemoryInterface {
  IndirectMemory(unsigned int table_size, std::string description = "")
      : description(description),
        nonstationary_table(table_size, 255),
        run_map_table(table_size, 0) {
    nonstationary_table.shrink_to_fit();
    run_map_table.shrink_to_fit();
  }
  std::string description;
  // Map from context to nonstationary state:
  std::vector<unsigned char> nonstationary_table;
  // Map from context to run map state:
  std::vector<unsigned char> run_map_table;
  // Map from state to prediction (in logit space).
  std::array<float, 256> nonstationary_predictions;
  // Map from state to prediction (in logit space).
  std::array<float, 256> run_map_predictions;

  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;
};

// This model maps contexts to two one byte "states" (see
// contexts/nonstationary.h and contexts/run-map.h). The state is then mapped to
// a probability. This model only supports contexts up to 24 bits.
class Indirect : public Model {
 public:
  // table_size: amount of memory to use for storing states. The context table
  // size will be 256 times larger than this value.
  // description: a short identifier for this model.
  Indirect(ShortTermMemory& short_term_memory, LongTermMemory& long_term_memory,
           float learning_rate, unsigned int table_size, unsigned int& context,
           std::string description, bool enable_analysis);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory) override;
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory) override;
  void WriteToDisk(std::ofstream* s) override {}
  void ReadFromDisk(std::ifstream* s) override {}
  void Copy(const MemoryInterface* m) override {}
  unsigned long long GetMemoryUsage(const ShortTermMemory& short_term_memory,
                                    const LongTermMemory& long_term_memory) override;

 private:
  IndirectMemory* GetMemory(LongTermMemory& long_term_memory);
  const IndirectMemory* GetMemory(const LongTermMemory& long_term_memory) const;

  unsigned int& context_;
  int prediction_index_indirect_, prediction_index_run_map_, memory_index_;
  float learning_rate_;
};

#endif  // MODELS_INDIRECT_H
