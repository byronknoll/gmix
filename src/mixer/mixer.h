#ifndef MIXER_MIXER_H_
#define MIXER_MIXER_H_

#include <memory>
#include <valarray>

#include "../model.h"
#include "sigmoid.h"

// A Mixer takes a set of predictions as input, along with a context. It uses
// context mixing to create a single output prediction.
class Mixer : public Model {
 public:
  // layer_number: 0: first layer, 1: second layer, 2: final layer
  // description: a short identifier for this mixer.
  Mixer(ShortTermMemory& short_term_memory, LongTermMemory& long_term_memory,
        unsigned int& context, const std::valarray<float>& inputs,
        float learning_rate, int layer_number, std::string description);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory);
  void WriteToDisk(std::ofstream* s);
  void ReadFromDisk(std::ifstream* s);
  void Copy(const MemoryInterface* m);
  unsigned long long GetMemoryUsage(const ShortTermMemory& short_term_memory,
                                    const LongTermMemory& long_term_memory);

 private:
  unsigned int& context_;
  unsigned long long max_steps_, steps_;
  int output_index_, memory_index_;
  float learning_rate_;
  const std::valarray<float>& inputs_;
  int layer_number_;

  MixerData* FindMixerData(const LongTermMemory& long_term_memory);
  MixerData* FindOrCreateMixerData(const ShortTermMemory& short_term_memory,
                                   LongTermMemory& long_term_memory);
};

#endif  // MIXER_MIXER_H_