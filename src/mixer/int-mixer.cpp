#include "int-mixer.h"

#include <algorithm>
#include <cmath>

IntMixerMemory::IntMixerMemory(unsigned int n_contexts, unsigned int n_inputs,
                               std::string desc)
    : description(desc),
      num_contexts(n_contexts),
      num_inputs(n_inputs),
      table(n_contexts * n_inputs, 0) {}

void IntMixerMemory::WriteToDisk(std::ofstream* s) {
  Serialize(s, num_contexts);
  Serialize(s, num_inputs);
  SerializeArray(s, table);
}

void IntMixerMemory::ReadFromDisk(std::ifstream* s) {
  Serialize(s, num_contexts);
  Serialize(s, num_inputs);
  SerializeArray(s, table);
}

void IntMixerMemory::Copy(const MemoryInterface* m) {
  const IntMixerMemory* orig = static_cast<const IntMixerMemory*>(m);
  description = orig->description;
  num_contexts = orig->num_contexts;
  num_inputs = orig->num_inputs;
  table = orig->table;
}

IntMixer::IntMixer(ShortTermMemory& short_term_memory,
                   LongTermMemory& long_term_memory,
                   const std::vector<int>& input_indices,
                   const unsigned int& context, unsigned int num_contexts,
                   float learning_rate, std::string description,
                   bool enable_analysis, bool add_skip_connection)
    : input_indices_(input_indices),
      context_(context),
      num_contexts_(num_contexts),
      learning_rate_(learning_rate),
      last_inputs_(input_indices.size(), 0) {
  prediction_index_ =
      short_term_memory.AddPrediction(description, enable_analysis, this);
  if (add_skip_connection) {
    short_term_memory.models_with_skip_connection.push_back(prediction_index_);
  }
  memory_index_ = long_term_memory.model_memory.size();
  long_term_memory.model_memory.push_back(std::make_unique<IntMixerMemory>(
      num_contexts_, static_cast<unsigned int>(input_indices_.size()),
      description));
}

IntMixerMemory* IntMixer::GetMemory(LongTermMemory& long_term_memory) {
  return static_cast<IntMixerMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

const IntMixerMemory* IntMixer::GetMemory(
    const LongTermMemory& long_term_memory) const {
  return static_cast<const IntMixerMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

void IntMixer::Predict(ShortTermMemory& short_term_memory,
                       const LongTermMemory& long_term_memory) {
  unsigned int c = (context_ ^ (context_ >> 16)) * 2654435761u +
                   (short_term_memory.bit_context * 2246822519u);
  c ^= (c >> 13);
  last_context_index_ = c & (num_contexts_ - 1);

  const auto& mem = *GetMemory(long_term_memory);
  const int16_t* w = &mem.table[last_context_index_ * input_indices_.size()];

  int64_t dot = 0;
  for (size_t i = 0; i < input_indices_.size(); ++i) {
    int idx = input_indices_[i];
    float logit = short_term_memory.predictions[idx];
    int val = static_cast<int>(logit * 256.0f);
    if (val < -2047) val = -2047;
    if (val > 2047) val = 2047;
    last_inputs_[i] = static_cast<int16_t>(val);
    dot += static_cast<int64_t>(last_inputs_[i]) * w[i];
  }

  float logit_out = static_cast<float>(dot) / 2097152.0f;
  float p = Sigmoid::Logistic(logit_out);
  last_prediction_ = p;
  short_term_memory.SetPrediction(p, prediction_index_);
}

void IntMixer::Learn(const ShortTermMemory& short_term_memory,
                     LongTermMemory& long_term_memory) {
  int y = short_term_memory.new_bit;
  float err = y - last_prediction_;
  float rate = learning_rate_;

  auto& mem = *GetMemory(long_term_memory);
  int16_t* w = &mem.table[last_context_index_ * input_indices_.size()];

  for (size_t i = 0; i < input_indices_.size(); ++i) {
    if (last_inputs_[i] == 0) continue;
    float delta = err * last_inputs_[i] * rate;
    int32_t new_w = static_cast<int32_t>(w[i]) + static_cast<int32_t>(delta);
    if (new_w < -32767) new_w = -32767;
    if (new_w > 32767) new_w = 32767;
    w[i] = static_cast<int16_t>(new_w);
  }
}

void IntMixer::WriteToDisk(std::ofstream* s) {
  Serialize(s, last_context_index_);
  Serialize(s, last_prediction_);
  SerializeArray(s, last_inputs_);
}

void IntMixer::ReadFromDisk(std::ifstream* s) {
  Serialize(s, last_context_index_);
  Serialize(s, last_prediction_);
  SerializeArray(s, last_inputs_);
}

void IntMixer::Copy(const MemoryInterface* m) {
  const IntMixer* orig = static_cast<const IntMixer*>(m);
  last_context_index_ = orig->last_context_index_;
  last_prediction_ = orig->last_prediction_;
  last_inputs_ = orig->last_inputs_;
}

unsigned long long IntMixer::GetMemoryUsage(
    const ShortTermMemory& short_term_memory,
    const LongTermMemory& long_term_memory) {
  const auto& mem = *GetMemory(long_term_memory);
  return sizeof(*this) + sizeof(IntMixerMemory) +
         mem.table.size() * sizeof(int16_t);
}
