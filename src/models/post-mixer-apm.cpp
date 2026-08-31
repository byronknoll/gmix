#include "post-mixer-apm.h"

#include <algorithm>
#include "../mixer/sigmoid.h"

PostMixerAPMMemory::PostMixerAPMMemory(unsigned int n_contexts, int n_bins,
                                       std::string desc)
    : description(desc), num_contexts(n_contexts), num_bins(n_bins) {
  table.resize(num_contexts * num_bins);
  float min_logit = -8.0f;
  float max_logit = 8.0f;
  float delta = (max_logit - min_logit) / (num_bins - 1);
  for (unsigned int c = 0; c < num_contexts; ++c) {
    for (int b = 0; b < num_bins; ++b) {
      table[c * num_bins + b] = min_logit + b * delta;
    }
  }
}

void PostMixerAPMMemory::WriteToDisk(std::ofstream* s) {
  Serialize(s, num_contexts);
  Serialize(s, num_bins);
  SerializeArray(s, table);
}

void PostMixerAPMMemory::ReadFromDisk(std::ifstream* s) {
  Serialize(s, num_contexts);
  Serialize(s, num_bins);
  SerializeArray(s, table);
}

void PostMixerAPMMemory::Copy(const MemoryInterface* m) {
  const PostMixerAPMMemory* orig = static_cast<const PostMixerAPMMemory*>(m);
  description = orig->description;
  num_contexts = orig->num_contexts;
  num_bins = orig->num_bins;
  table = orig->table;
}

PostMixerAPM::PostMixerAPM(ShortTermMemory& short_term_memory,
                           LongTermMemory& long_term_memory,
                           const unsigned int& context,
                           unsigned int num_contexts,
                           float learning_rate,
                           float blend_weight,
                           std::string description)
    : context_(context),
      num_contexts_(num_contexts),
      learning_rate_(learning_rate),
      blend_weight_(blend_weight) {
  memory_index_ = long_term_memory.model_memory.size();
  long_term_memory.model_memory.push_back(
      std::make_unique<PostMixerAPMMemory>(num_contexts_, kNumBins, description));
}

PostMixerAPMMemory* PostMixerAPM::GetMemory(LongTermMemory& long_term_memory) {
  return static_cast<PostMixerAPMMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

const PostMixerAPMMemory* PostMixerAPM::GetMemory(
    const LongTermMemory& long_term_memory) const {
  return static_cast<const PostMixerAPMMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

void PostMixerAPM::Predict(ShortTermMemory& short_term_memory,
                           const LongTermMemory& long_term_memory) {
  float x = short_term_memory.final_mixer_output;
  if (x < kMinLogit + 0.01f) x = kMinLogit + 0.01f;
  if (x > kMaxLogit - 0.01f) x = kMaxLogit - 0.01f;

  float u = (x - kMinLogit) / kBinDelta;
  int b = static_cast<int>(u);
  if (b < 0) b = 0;
  if (b > kNumBins - 2) b = kNumBins - 2;
  float w = u - b;

  unsigned int c = context_ % num_contexts_;
  unsigned int idx = c * kNumBins + b;

  const auto& mem = *GetMemory(long_term_memory);
  float out = (1.0f - w) * mem.table[idx] + w * mem.table[idx + 1];

  last_context_ = c;
  last_bin_ = b;
  last_weight_ = w;
  last_output_ = out;

  short_term_memory.final_mixer_output =
      (1.0f - blend_weight_) * short_term_memory.final_mixer_output +
      blend_weight_ * out;
}

void PostMixerAPM::Learn(const ShortTermMemory& short_term_memory,
                         LongTermMemory& long_term_memory) {
  int y = short_term_memory.new_bit;
  float p = Sigmoid::Logistic(last_output_);
  float err = y - p;
  float update = learning_rate_ * err;

  auto& mem = *GetMemory(long_term_memory);
  unsigned int idx = last_context_ * kNumBins + last_bin_;
  mem.table[idx] += (1.0f - last_weight_) * update;
  mem.table[idx + 1] += last_weight_ * update;
}

void PostMixerAPM::WriteToDisk(std::ofstream* s) {
  Serialize(s, last_context_);
  Serialize(s, last_bin_);
  Serialize(s, last_weight_);
  Serialize(s, last_output_);
}

void PostMixerAPM::ReadFromDisk(std::ifstream* s) {
  Serialize(s, last_context_);
  Serialize(s, last_bin_);
  Serialize(s, last_weight_);
  Serialize(s, last_output_);
}

void PostMixerAPM::Copy(const MemoryInterface* m) {
  const PostMixerAPM* orig = static_cast<const PostMixerAPM*>(m);
  last_context_ = orig->last_context_;
  last_bin_ = orig->last_bin_;
  last_weight_ = orig->last_weight_;
  last_output_ = orig->last_output_;
}

unsigned long long PostMixerAPM::GetMemoryUsage(
    const ShortTermMemory& short_term_memory,
    const LongTermMemory& long_term_memory) {
  unsigned long long usage = sizeof(*this);
  usage += GetMemory(long_term_memory)->table.size() * sizeof(float);
  return usage;
}
