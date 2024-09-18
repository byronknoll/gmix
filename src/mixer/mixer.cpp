#include "mixer.h"

Mixer::Mixer(ShortTermMemory& short_term_memory, unsigned long long& context,
             const Sigmoid& sigmoid, float learning_rate)
    : context_(context),
      sigmoid_(sigmoid),
      max_steps_(1),
      steps_(0),
      learning_rate_(learning_rate) {}

MixerData* Mixer::FindMixerData(const LongTermMemory& long_term_memory) {
  MixerData* data = nullptr;
  if (long_term_memory.mixer_map.find(context_) ==
      long_term_memory.mixer_map.end()) {
    unsigned long long limit = 10000;
    if (long_term_memory.mixer_map.size() >= limit &&
        long_term_memory.mixer_map.find(0xDEADBEEF) !=
            long_term_memory.mixer_map.end()) {
      data = long_term_memory.mixer_map.at(0xDEADBEEF).get();
    }
  } else {
    data = long_term_memory.mixer_map.at(context_).get();
  }

  return data;
}

MixerData* Mixer::FindOrCreateMixerData(
    const ShortTermMemory& short_term_memory,
    LongTermMemory& long_term_memory) {
  MixerData* data;
  unsigned long long limit = 10000;
  if (long_term_memory.mixer_map.size() >= limit &&
      long_term_memory.mixer_map.find(context_) ==
          long_term_memory.mixer_map.end()) {
    data = long_term_memory.mixer_map[0xDEADBEEF].get();
    if (data == nullptr) {
      long_term_memory.mixer_map[0xDEADBEEF] = std::unique_ptr<MixerData>(
          new MixerData(short_term_memory.predictions.size()));
      data = long_term_memory.mixer_map[0xDEADBEEF].get();
    }
  } else {
    data = long_term_memory.mixer_map[context_].get();
    if (data == nullptr) {
      long_term_memory.mixer_map[context_] = std::unique_ptr<MixerData>(
          new MixerData(short_term_memory.predictions.size()));
      data = long_term_memory.mixer_map[context_].get();
    }
  }

  return data;
}

void Mixer::Predict(ShortTermMemory& short_term_memory,
                    const LongTermMemory& long_term_memory) {
  MixerData* data = FindMixerData(long_term_memory);
  float p = 0;
  if (data != nullptr) {
    for (int i = 0; i < short_term_memory.predictions.size(); ++i) {
      p += sigmoid_.Logit(short_term_memory.predictions[i]) * data->weights[i];
    }
  }
  short_term_memory.mixer_output = Sigmoid::Logistic(p);
}

void Mixer::Learn(const ShortTermMemory& short_term_memory,
                  LongTermMemory& long_term_memory) {
  MixerData* data = FindOrCreateMixerData(short_term_memory, long_term_memory);
  float decay = 0.9 / pow(0.0000001 * steps_ + 0.8, 0.8);
  decay *= 1.5 - ((1.0 * data->steps) / max_steps_);
  float update = decay * learning_rate_ *
                 (short_term_memory.mixer_output - short_term_memory.new_bit);
  ++steps_;
  ++data->steps;
  if (data->steps > max_steps_) {
    max_steps_ = data->steps;
  }
  data->weights -= update * short_term_memory.predictions;
  if ((data->steps & 1023) == 0) {
    data->weights *= 1.0f - 3.0e-6f;
  }
}