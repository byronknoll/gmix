#ifndef MIXER_MIXER_H_
#define MIXER_MIXER_H_

#include <memory>

#include "../model.h"
#include "sigmoid.h"

class Mixer : public Model {
 public:
  Mixer(ShortTermMemory& short_term_memory, unsigned long long& context,
        const Sigmoid& sigmoid, float learning_rate);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Perceive(ShortTermMemory& short_term_memory,
                const LongTermMemory& long_term_memory) {}
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory);
  void WriteToDisk() {}
  void ReadFromDisk() {}

 private:
  unsigned long long& context_;
  const Sigmoid& sigmoid_;
  unsigned long long max_steps_, steps_;
  float learning_rate_;

  MixerData* FindMixerData(const LongTermMemory& long_term_memory);
  MixerData* FindOrCreateMixerData(const ShortTermMemory& short_term_memory,
                                   LongTermMemory& long_term_memory);
};

#endif  // MIXER_MIXER_H_