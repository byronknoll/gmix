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
  // table_size: the size of the table used for storing context matches.
  // description: a short identifier for this mixer.
  Mixer(ShortTermMemory& short_term_memory, LongTermMemory& long_term_memory,
        unsigned int& context, float learning_rate, int layer_number,
        unsigned long long table_size, std::string description,
        bool enable_analysis);
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
  // Each context is keeps track of how many times it is seen. max_steps_ is the
  // max of those.
  unsigned long long max_steps_;
  // steps_ is the number of times "Learn" has been called.
  unsigned long long steps_;
  // contexts_seen_ is the number of unique contexts which have been seen.
  unsigned long long contexts_seen_ = 0;
  int output_index_, memory_index_, weight_size_;
  float learning_rate_;
  int layer_number_;

  MixerData* FindMixerData(const LongTermMemory& long_term_memory);
  MixerData* FindOrCreateMixerData(const ShortTermMemory& short_term_memory,
                                   LongTermMemory& long_term_memory);
};

#endif  // MIXER_MIXER_H_