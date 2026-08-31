#ifndef APM_H_
#define APM_H_

#include <string>
#include <vector>

#include "../model.h"

struct APMMemory : public MemoryInterface {
 public:
  APMMemory(unsigned int num_contexts, int num_bins, std::string description);
  ~APMMemory() {}
  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;

  std::string description;
  unsigned int num_contexts = 0;
  int num_bins = 33;
  std::vector<float> table;
};

// Adaptive Probability Map (APM) adaptively calibrates a model's prediction
// conditioned on a given context.
class APM : public Model {
 public:
  APM(ShortTermMemory& short_term_memory,
      LongTermMemory& long_term_memory,
      int input_prediction_index,
      const unsigned int& context,
      unsigned int num_contexts,
      float learning_rate,
      std::string description,
      bool enable_analysis);
  ~APM() {}

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

 private:
  int input_prediction_index_;
  int output_prediction_index_;
  const unsigned int& context_;
  unsigned int num_contexts_;
  float learning_rate_;
  int memory_index_;

  // Short-term state for current prediction.
  unsigned int last_context_ = 0;
  int last_bin_ = 0;
  float last_weight_ = 0.0f;
  float last_output_ = 0.0f;

  static constexpr int kNumBins = 33;
  static constexpr float kMinLogit = -8.0f;
  static constexpr float kMaxLogit = 8.0f;
  static constexpr float kBinDelta = (kMaxLogit - kMinLogit) / (kNumBins - 1);

  APMMemory* GetMemory(LongTermMemory& long_term_memory);
  const APMMemory* GetMemory(const LongTermMemory& long_term_memory) const;
};

#endif  // APM_H_
