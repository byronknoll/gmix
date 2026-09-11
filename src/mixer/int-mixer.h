#ifndef MIXER_INT_MIXER_H_
#define MIXER_INT_MIXER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../model.h"
#include "sigmoid.h"

struct IntMixerMemory : public MemoryInterface {
  IntMixerMemory(unsigned int n_contexts, unsigned int n_inputs,
                 std::string desc = "");
  std::string description;
  unsigned int num_contexts;
  unsigned int num_inputs;
  std::vector<int16_t> table;

  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;
};

// IntMixer is a fast, memory-efficient 16-bit fixed-point mixer network.
// It can mix a specific cluster of models (e.g. word models, match models)
// using integer dot-products and adapt via fixed-point gradient descent.
class IntMixer : public Model {
 public:
  IntMixer(ShortTermMemory& short_term_memory, LongTermMemory& long_term_memory,
           const std::vector<int>& input_indices, const unsigned int& context,
           unsigned int num_contexts, float learning_rate,
           std::string description, bool enable_analysis = false,
           bool add_skip_connection = true);

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

  int GetPredictionIndex() const { return prediction_index_; }

 private:
  IntMixerMemory* GetMemory(LongTermMemory& long_term_memory);
  const IntMixerMemory* GetMemory(const LongTermMemory& long_term_memory) const;

  std::vector<int> input_indices_;
  const unsigned int& context_;
  unsigned int num_contexts_;
  float learning_rate_;
  int prediction_index_;
  int memory_index_;

  unsigned int last_context_index_ = 0;
  float last_prediction_ = 0.5f;
  std::vector<int16_t> last_inputs_;
};

#endif  // MIXER_INT_MIXER_H_
