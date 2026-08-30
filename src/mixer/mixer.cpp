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
  MixerData* data = nullptr;
  auto& mixer_table = GetMemory(long_term_memory)->mixer_table;
  auto& ptr = mixer_table[context_ % mixer_table.size()];
  if (ptr) {
    data = ptr.get();
  }
  return data;
}

MixerData* Mixer::FindOrCreateMixerData(
    const ShortTermMemory& short_term_memory,
    LongTermMemory& long_term_memory) {
  auto& mixer_table = GetMemory(long_term_memory)->mixer_table;
  auto& ptr = mixer_table[context_ % mixer_table.size()];
  if (!ptr) {
    ++contexts_seen_;
    ptr.reset(new MixerData(weight_size_));
  }
  return ptr.get();
}

void Mixer::Predict(ShortTermMemory& short_term_memory,
                    const LongTermMemory& long_term_memory) {
  MixerData* data = FindMixerData(long_term_memory);
  float p = 0;
  if (data != nullptr) {
    if (layer_number_ == 0) {
      for (int i : short_term_memory.active_models) {
        p += short_term_memory.predictions[i] * data->weights[i];
      }
      // Use the previous mixers in the same layer.
      for (int i = 0; i < output_index_; ++i) {
        p += short_term_memory.mixer_layer0_outputs[i] *
             data->weights[short_term_memory.num_predictions + i];
      }
    } else if (layer_number_ == 1) {
      for (int i = 0; i < short_term_memory.num_layer0_mixers; ++i) {
        p += short_term_memory.mixer_layer0_outputs[i] * data->weights[i];
      }
      // Use the previous mixers in the same layer.
      for (int i = 0; i < output_index_; ++i) {
        p += short_term_memory.mixer_layer1_outputs[i] *
             data->weights[short_term_memory.num_layer0_mixers + i];
      }
      // Skip connections.
      int offset = short_term_memory.num_layer0_mixers + output_index_;
      for (int i = 0; i < short_term_memory.models_with_skip_connection.size();
           ++i) {
        int index = short_term_memory.models_with_skip_connection[i];
        p += short_term_memory.predictions[index] * data->weights[offset + i];
      }
    } else {
      for (int i = 0; i < short_term_memory.num_layer0_mixers; ++i) {
        p += short_term_memory.mixer_layer0_outputs[i] * data->weights[i];
      }
      for (int i = 0; i < short_term_memory.num_layer1_mixers; ++i) {
        p += short_term_memory.mixer_layer1_outputs[i] *
             data->weights[short_term_memory.num_layer0_mixers + i];
      }
      // Skip connections.
      int offset = short_term_memory.num_layer0_mixers +
                   short_term_memory.num_layer1_mixers;
      for (int i = 0; i < short_term_memory.models_with_skip_connection.size();
           ++i) {
        int index = short_term_memory.models_with_skip_connection[i];
        p += short_term_memory.predictions[index] * data->weights[offset + i];
      }
    }
  }
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
  MixerData* data = FindOrCreateMixerData(short_term_memory, long_term_memory);
  if (data->steps > max_steps_) {
    max_steps_ = data->steps;
  }
  if (steps_ < max_steps_) {
    steps_ = max_steps_;
  }
  float decay = 0.9 / pow(0.0000001 * steps_ + 0.8, 0.8);
  decay *= 1.5 - ((1.0 * data->steps) / max_steps_);
  float p;
  if (layer_number_ == 2) {
    p = Sigmoid::Logistic(short_term_memory.final_mixer_output);
  } else if (layer_number_ == 1) {
    p = Sigmoid::Logistic(
        short_term_memory.mixer_layer1_outputs[output_index_]);
  } else {
    p = Sigmoid::Logistic(
        short_term_memory.mixer_layer0_outputs[output_index_]);
  }
  float update = decay * learning_rate_ * (p - short_term_memory.new_bit);
  ++steps_;
  ++data->steps;
  if (data->steps > max_steps_) {
    max_steps_ = data->steps;
  }
  if (layer_number_ == 0) {
    for (int i : short_term_memory.active_models) {
      data->weights[i] -= update * short_term_memory.predictions[i];
    }
    // Use the previous mixers in the same layer.
    for (int i = 0; i < output_index_; ++i) {
      data->weights[i + short_term_memory.num_predictions] -=
          update * short_term_memory.mixer_layer0_outputs[i];
    }
  } else if (layer_number_ == 1) {
    for (int i = 0; i < short_term_memory.num_layer0_mixers; ++i) {
      data->weights[i] -= update * short_term_memory.mixer_layer0_outputs[i];
    }
    // Use the previous mixers in the same layer.
    for (int i = 0; i < output_index_; ++i) {
      data->weights[i + short_term_memory.num_layer0_mixers] -=
          update * short_term_memory.mixer_layer1_outputs[i];
    }
    // Skip connections.
    int offset = short_term_memory.num_layer0_mixers + output_index_;
    for (int i = 0; i < short_term_memory.models_with_skip_connection.size();
         ++i) {
      int index = short_term_memory.models_with_skip_connection[i];
      data->weights[i + offset] -=
          update * short_term_memory.predictions[index];
    }
  } else {
    for (int i = 0; i < short_term_memory.num_layer0_mixers; ++i) {
      data->weights[i] -= update * short_term_memory.mixer_layer0_outputs[i];
    }
    for (int i = 0; i < short_term_memory.num_layer1_mixers; ++i) {
      data->weights[short_term_memory.num_layer0_mixers + i] -=
          update * short_term_memory.mixer_layer1_outputs[i];
    }
    // Skip connections.
    int offset = short_term_memory.num_layer0_mixers +
                 short_term_memory.num_layer1_mixers;
    for (int i = 0; i < short_term_memory.models_with_skip_connection.size();
         ++i) {
      int index = short_term_memory.models_with_skip_connection[i];
      data->weights[i + offset] -=
          update * short_term_memory.predictions[index];
    }
  }
  if ((data->steps & 1023) == 0) {
    data->weights *= 1.0f - 3.0e-6f;  // Weight regularization.
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
}

void Mixer::Copy(const MemoryInterface* m) {
  const Mixer* orig = static_cast<const Mixer*>(m);
  steps_ = orig->steps_;
  max_steps_ = orig->max_steps_;
  contexts_seen_ = orig->contexts_seen_;
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