#include "mixer.h"

Mixer::Mixer(ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory, unsigned int& context,
             float learning_rate, int layer_number, unsigned int table_size,
             std::string description, bool enable_analysis)
    : context_(context),
      max_steps_(1),
      steps_(0),
      learning_rate_(learning_rate),
      layer_number_(layer_number) {
  output_index_ = short_term_memory.AddMixer(description, layer_number,
                                             enable_analysis, this);
  memory_index_ = long_term_memory.model_memory.size();
  long_term_memory.model_memory.push_back(
      std::make_unique<MixerMemory>(table_size, description));

  if (layer_number_ == 0) {
    weight_size_ = short_term_memory.num_predictions + output_index_;
  } else if (layer_number_ == 1) {
    weight_size_ = short_term_memory.num_layer0_mixers + output_index_ +
                   short_term_memory.models_with_skip_connection.size();
  } else {
    weight_size_ = short_term_memory.num_layer0_mixers +
                   short_term_memory.num_layer1_mixers +
                   short_term_memory.models_with_skip_connection.size();
  }
  if ((table_size & (table_size - 1)) == 0) {
    mask_ = table_size - 1;
  }
}

MixerMemory* Mixer::GetMemory(LongTermMemory& long_term_memory) {
  return static_cast<MixerMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

const MixerMemory* Mixer::GetMemory(
    const LongTermMemory& long_term_memory) const {
  return static_cast<const MixerMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

MixerData* Mixer::FindMixerData(const LongTermMemory& long_term_memory) {
  auto& mixer_table = GetMemory(long_term_memory)->mixer_table;
  unsigned int idx = mask_ ? (context_ & mask_) : (context_ % mixer_table.size());
  auto& ptr = mixer_table[idx];
  return ptr ? ptr.get() : nullptr;
}

MixerData* Mixer::FindOrCreateMixerData(
    const ShortTermMemory& short_term_memory,
    LongTermMemory& long_term_memory) {
  auto& mixer_table = GetMemory(long_term_memory)->mixer_table;
  unsigned int idx = mask_ ? (context_ & mask_) : (context_ % mixer_table.size());
  auto& ptr = mixer_table[idx];
  if (!ptr) {
    ++contexts_seen_;
    ptr.reset(new MixerData(weight_size_));
  }
  return ptr.get();
}

void Mixer::Predict(ShortTermMemory& short_term_memory,
                    const LongTermMemory& long_term_memory) {
  auto& mixer_table = GetMemory(const_cast<LongTermMemory&>(long_term_memory))->mixer_table;
  last_idx_ = mask_ ? (context_ & mask_) : (context_ % mixer_table.size());
  auto& ptr = mixer_table[last_idx_];
  MixerData* data = ptr ? ptr.get() : nullptr;
  last_mixer_data_ = data;
  float p = 0;
  if (data != nullptr) {
    const float* __restrict__ const w = data->weights.data();
    if (layer_number_ == 0) {
      const float* __restrict__ const preds = &short_term_memory.predictions[0];
      const int num_preds = short_term_memory.num_predictions;
      for (int i = 0; i < num_preds; ++i) {
        p += preds[i] * w[i];
      }
      const float* __restrict__ const m0 = &short_term_memory.mixer_layer0_outputs[0];
      for (int i = 0; i < output_index_; ++i) {
        p += m0[i] * w[num_preds + i];
      }
    } else if (layer_number_ == 1) {
      const float* __restrict__ const m0 = &short_term_memory.mixer_layer0_outputs[0];
      const int m0_size = short_term_memory.num_layer0_mixers;
      for (int i = 0; i < m0_size; ++i) {
        p += m0[i] * w[i];
      }
      const float* __restrict__ const m1 = &short_term_memory.mixer_layer1_outputs[0];
      for (int i = 0; i < output_index_; ++i) {
        p += m1[i] * w[m0_size + i];
      }
      int offset = m0_size + output_index_;
      const float* __restrict__ const preds = &short_term_memory.predictions[0];
      for (int index : short_term_memory.models_with_skip_connection) {
        p += preds[index] * w[offset++];
      }
    } else {
      const float* __restrict__ const m0 = &short_term_memory.mixer_layer0_outputs[0];
      const int m0_size = short_term_memory.num_layer0_mixers;
      for (int i = 0; i < m0_size; ++i) {
        p += m0[i] * w[i];
      }
      const float* __restrict__ const m1 = &short_term_memory.mixer_layer1_outputs[0];
      const int m1_size = short_term_memory.num_layer1_mixers;
      for (int i = 0; i < m1_size; ++i) {
        p += m1[i] * w[m0_size + i];
      }
      int offset = m0_size + m1_size;
      const float* __restrict__ const preds = &short_term_memory.predictions[0];
      for (int index : short_term_memory.models_with_skip_connection) {
        p += preds[index] * w[offset++];
      }
    }
  }
  last_output_ = p;
  if (layer_number_ == 2) {
    short_term_memory.final_mixer_output = p;
    // printf("Mixer2 output = %f\n", p);
  } else if (layer_number_ == 1) {
    short_term_memory.mixer_layer1_outputs[output_index_] = p;
  } else {
    short_term_memory.mixer_layer0_outputs[output_index_] = p;
  }
}

void Mixer::Learn(const ShortTermMemory& short_term_memory,
                  LongTermMemory& long_term_memory) {
  MixerData* data = last_mixer_data_;
  if (!data) {
    auto& mixer_table = GetMemory(long_term_memory)->mixer_table;
    auto& ptr = mixer_table[last_idx_];
    if (!ptr) {
      ++contexts_seen_;
      ptr.reset(new MixerData(weight_size_));
    }
    data = ptr.get();
  }
  if (data->steps > max_steps_) {
    max_steps_ = data->steps;
  }
  if (steps_ < max_steps_) {
    steps_ = max_steps_;
  }
  if ((steps_ & 15) == 0) {
    cached_decay_ = 0.9f / std::pow(0.0000001f * steps_ + 0.8f, 0.8f);
  }
  float decay = cached_decay_ * (1.5f - ((1.0f * data->steps) / max_steps_));
  float p = Sigmoid::Logistic(last_output_);
  float update = decay * learning_rate_ * (p - short_term_memory.new_bit);
  ++steps_;
  ++data->steps;
  if (data->steps > max_steps_) {
    max_steps_ = data->steps;
  }
  float* __restrict__ const w = data->weights.data();
  if (layer_number_ == 0) {
    const float* __restrict__ const preds = &short_term_memory.predictions[0];
    const int num_preds = short_term_memory.num_predictions;
    for (int i = 0; i < num_preds; ++i) {
      w[i] -= update * preds[i];
    }
    const float* __restrict__ const m0 = &short_term_memory.mixer_layer0_outputs[0];
    for (int i = 0; i < output_index_; ++i) {
      w[num_preds + i] -= update * m0[i];
    }
  } else if (layer_number_ == 1) {
    const float* __restrict__ const m0 = &short_term_memory.mixer_layer0_outputs[0];
    const int m0_size = short_term_memory.num_layer0_mixers;
    for (int i = 0; i < m0_size; ++i) {
      w[i] -= update * m0[i];
    }
    const float* __restrict__ const m1 = &short_term_memory.mixer_layer1_outputs[0];
    for (int i = 0; i < output_index_; ++i) {
      w[m0_size + i] -= update * m1[i];
    }
    int offset = m0_size + output_index_;
    const float* __restrict__ const preds = &short_term_memory.predictions[0];
    for (int index : short_term_memory.models_with_skip_connection) {
      w[offset++] -= update * preds[index];
    }
  } else {
    const float* __restrict__ const m0 = &short_term_memory.mixer_layer0_outputs[0];
    const int m0_size = short_term_memory.num_layer0_mixers;
    for (int i = 0; i < m0_size; ++i) {
      w[i] -= update * m0[i];
    }
    const float* __restrict__ const m1 = &short_term_memory.mixer_layer1_outputs[0];
    const int m1_size = short_term_memory.num_layer1_mixers;
    for (int i = 0; i < m1_size; ++i) {
      w[m0_size + i] -= update * m1[i];
    }
    int offset = m0_size + m1_size;
    const float* __restrict__ const preds = &short_term_memory.predictions[0];
    for (int index : short_term_memory.models_with_skip_connection) {
      w[offset++] -= update * preds[index];
    }
  }
  if ((data->steps & 1023) == 0) {
    const float factor = 1.0f - 3.0e-6f;
    for (int i = 0; i < weight_size_; ++i) {
      w[i] *= factor;
    }
  }
}

void Mixer::WriteToDisk(std::ofstream* s) {
  Serialize(s, steps_);
  Serialize(s, max_steps_);
  Serialize(s, contexts_seen_);
}

void Mixer::ReadFromDisk(std::ifstream* s) {
  Serialize(s, steps_);
  Serialize(s, max_steps_);
  Serialize(s, contexts_seen_);
  cached_decay_ = 0.9f / std::pow(0.0000001f * steps_ + 0.8f, 0.8f);
}

void Mixer::Copy(const MemoryInterface* m) {
  const Mixer* orig = static_cast<const Mixer*>(m);
  steps_ = orig->steps_;
  max_steps_ = orig->max_steps_;
  contexts_seen_ = orig->contexts_seen_;
  cached_decay_ = orig->cached_decay_;
  last_output_ = orig->last_output_;
}

unsigned long long Mixer::GetMemoryUsage(
    const ShortTermMemory& short_term_memory,
    const LongTermMemory& long_term_memory) {
  unsigned long long usage = 29;
  int mixer_data_size = weight_size_ * 4 + 12;
  usage += contexts_seen_ * mixer_data_size;
  auto& mixer_table = GetMemory(long_term_memory)->mixer_table;
  usage += 8 * mixer_table.size();
  return usage;
}

void MixerMemory::WriteToDisk(std::ofstream* s) {
  unsigned int input_size = 0;
  std::vector<unsigned int> keys;
  for (int i = 0; i < mixer_table.size(); ++i) {
    auto& ptr = mixer_table[i];
    if (ptr) {
      keys.push_back(i);
      input_size = ptr->weights.size();
    }
  }
  unsigned int mixer_size = keys.size();
  Serialize(s, mixer_size);
  Serialize(s, input_size);
  for (unsigned int context : keys) {
    Serialize(s, context);
    Serialize(s, mixer_table[context]->steps);
    SerializeArray(s, mixer_table[context]->weights);
  }
}

void MixerMemory::ReadFromDisk(std::ifstream* s) {
  unsigned int mixer_size = mixer_table.size();
  mixer_table.clear();
  mixer_table.resize(mixer_size);
  mixer_table.shrink_to_fit();
  Serialize(s, mixer_size);
  unsigned int input_size;
  Serialize(s, input_size);
  for (int i = 0; i < mixer_size; ++i) {
    unsigned int context;
    Serialize(s, context);
    mixer_table[context].reset(new MixerData(input_size));
    Serialize(s, mixer_table[context]->steps);
    SerializeArray(s, mixer_table[context]->weights);
  }
}

void MixerMemory::Copy(const MemoryInterface* m) {
  const MixerMemory* orig = static_cast<const MixerMemory*>(m);
  description = orig->description;
  mixer_table.clear();
  mixer_table.resize(orig->mixer_table.size());
  mixer_table.shrink_to_fit();
  for (int j = 0; j < orig->mixer_table.size(); ++j) {
    auto& orig_ptr = orig->mixer_table[j];
    if (orig_ptr) {
      auto& ptr = mixer_table[j];
      ptr.reset(new MixerData(orig_ptr->weights.size()));
      ptr->steps = orig_ptr->steps;
      ptr->weights = orig_ptr->weights;
    }
  }
}