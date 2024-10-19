#include "mixer.h"

Mixer::Mixer(ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory, unsigned int& context,
             const std::valarray<float>& inputs, float learning_rate,
             bool final_layer, std::string description)
    : context_(context),
      max_steps_(1),
      steps_(0),
      learning_rate_(learning_rate),
      inputs_(inputs),
      final_layer_(final_layer) {
  output_index_ = short_term_memory.AddMixer(description, this);
  memory_index_ = long_term_memory.mixers.size();
  long_term_memory.mixers.push_back(MixerMemory());
}

MixerData* Mixer::FindMixerData(const LongTermMemory& long_term_memory) {
  MixerData* data = nullptr;
  auto& mixer_map = long_term_memory.mixers[memory_index_].mixer_map;
  if (mixer_map.find(context_) != mixer_map.end()) {
    data = mixer_map.at(context_).get();
  }

  return data;
}

MixerData* Mixer::FindOrCreateMixerData(
    const ShortTermMemory& short_term_memory,
    LongTermMemory& long_term_memory) {
  auto& mixer_map = long_term_memory.mixers[memory_index_].mixer_map;
  MixerData* data = mixer_map[context_].get();
  if (data == nullptr) {
    mixer_map[context_] =
        std::unique_ptr<MixerData>(new MixerData(inputs_.size()));
    data = mixer_map[context_].get();
  }

  return data;
}

void Mixer::Predict(ShortTermMemory& short_term_memory,
                    const LongTermMemory& long_term_memory) {
  MixerData* data = FindMixerData(long_term_memory);
  float p = 0;
  if (data != nullptr) {
    if (final_layer_) {
      for (int i = 0; i < inputs_.size(); ++i) {
        p += inputs_[i] * data->weights[i];
      }
    } else {
      for (int i : short_term_memory.active_models) {
        p += inputs_[i] * data->weights[i];
      }
    }
  }
  if (final_layer_) {
    short_term_memory.final_mixer_output = p;
  } else {
    short_term_memory.mixer_outputs[output_index_] = p;
  }
}

void Mixer::Learn(const ShortTermMemory& short_term_memory,
                  LongTermMemory& long_term_memory) {
  MixerData* data = FindOrCreateMixerData(short_term_memory, long_term_memory);
  float decay = 0.9 / pow(0.0000001 * steps_ + 0.8, 0.8);
  decay *= 1.5 - ((1.0 * data->steps) / max_steps_);
  float p;
  if (final_layer_) {
    p = Sigmoid::Logistic(short_term_memory.final_mixer_output);
  } else {
    p = Sigmoid::Logistic(short_term_memory.mixer_outputs[output_index_]);
  }
  float update = decay * learning_rate_ * (p - short_term_memory.new_bit);
  ++steps_;
  ++data->steps;
  if (data->steps > max_steps_) {
    max_steps_ = data->steps;
  }
  if (final_layer_) {
    data->weights -= update * inputs_;
  } else {
    for (int i : short_term_memory.active_models) {
      data->weights[i] -= update * inputs_[i];
    }
  }
  if ((data->steps & 1023) == 0) {
    data->weights *= 1.0f - 3.0e-6f;  // Weight regularization.
  }
}

void Mixer::WriteToDisk(std::ofstream* s) {
  Serialize(s, steps_);
  Serialize(s, max_steps_);
}

void Mixer::ReadFromDisk(std::ifstream* s) {
  Serialize(s, steps_);
  Serialize(s, max_steps_);
}

void Mixer::Copy(const MemoryInterface* m) {
  const Mixer* orig = static_cast<const Mixer*>(m);
  steps_ = orig->steps_;
  max_steps_ = orig->max_steps_;
}

unsigned long long Mixer::GetMemoryUsage(
    const ShortTermMemory& short_term_memory,
    const LongTermMemory& long_term_memory) {
  unsigned long long usage = 29;
  int mixer_data_size = inputs_.size() * 4 + 12;
  auto& mixer_map = long_term_memory.mixers[memory_index_].mixer_map;
  usage += mixer_map.size() * mixer_data_size;
  return usage;
}