#ifndef POST_MIXER_APM_H_
#define POST_MIXER_APM_H_

#include <string>
#include <vector>

#include "../model.h"

struct PostMixerAPMMemory : public MemoryInterface {
 public:
  PostMixerAPMMemory(unsigned int num_contexts, int num_bins,
                     std::string description);
  ~PostMixerAPMMemory() {}
  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;

  std::string description;
  unsigned int num_contexts = 0;
  int num_bins = 33;
  std::vector<float> table;
};

// PostMixerAPM applies secondary symbol estimation (SSE) on the final mixer
// output conditioned on a given context.
class PostMixerAPM : public Model {
 public:
  PostMixerAPM(ShortTermMemory& short_term_memory,
               LongTermMemory& long_term_memory,
               const unsigned int& context,
               unsigned int num_contexts,
               float learning_rate,
               float blend_weight,
               std::string description);
  ~PostMixerAPM() {}

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
  const unsigned int& context_;
  unsigned int num_contexts_;
  float learning_rate_;
  float blend_weight_;
  int memory_index_;

  // Short-term state.
  unsigned int last_context_ = 0;
  int last_bin_ = 0;
  float last_weight_ = 0.0f;
  float last_output_ = 0.0f;

  static constexpr int kNumBins = 33;
  static constexpr float kMinLogit = -8.0f;
  static constexpr float kMaxLogit = 8.0f;
  static constexpr float kBinDelta = (kMaxLogit - kMinLogit) / (kNumBins - 1);

  PostMixerAPMMemory* GetMemory(LongTermMemory& long_term_memory);
  const PostMixerAPMMemory* GetMemory(
      const LongTermMemory& long_term_memory) const;
};

#endif  // POST_MIXER_APM_H_
