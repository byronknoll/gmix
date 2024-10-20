#ifndef SHORT_TERM_MEMORY_H_
#define SHORT_TERM_MEMORY_H_

#include <valarray>
#include <vector>
#include <unordered_map>

#include "../memory-interface.h"
#include "../mixer/sigmoid.h"
#include "../contexts/nonstationary.h"

class Model;

// ShortTermMemory contains "state" models need in order to make predictions,
// but does not contain any data used for training/learning. Models can also
// store state within member variables of their class, so the primary purpose of
// this struct is as a way to share inputs/outputs between models.
struct ShortTermMemory : MemoryInterface {
 public:
  ShortTermMemory(const Sigmoid& sigmoid)
      : ppm_predictions(1.0 / 256, 256), sigmoid(sigmoid) {}
  ~ShortTermMemory() {}
  void WriteToDisk(std::ofstream* s);
  void ReadFromDisk(std::ifstream* s);
  void Copy(const MemoryInterface* m);

  // Models should call this in their constructor.
  // description: a short identifier for this model.
  // ptr: a pointer to this model.
  // returns: model prediction index.
  int AddPrediction(std::string description, Model* ptr) {
    ++num_predictions;
    model_descriptions.push_back(description);
    prediction_index_to_model_ptr[num_predictions - 1] = ptr;
    return num_predictions - 1;
  }
  // Predictions for the next bit of data. Each prediction should be a
  // probability between 0 to 1.
  void SetPrediction(float prediction, int index) {
    predictions[index] = sigmoid.Logit(prediction);
    active_models.push_back(index);
  }
  std::valarray<float> predictions;
  std::vector<int> active_models;
   int num_predictions = 0;
  std::vector<std::string> model_descriptions;
  std::unordered_map<int, Model*> prediction_index_to_model_ptr;

  // The most recently perceived bit.
  int new_bit = 0;

  // Recently perceived bits of the current byte. The leftmost "1" bit
  // indicates how many bits have been seen. Ranges in value from 1 to 255.
  // 1: 0 bits seen
  // 2-3: 1 bit seen: (2=zero, 3=one)
  // ...
  // 128-255: 7 bits seen
  // This gets updated *after* the "Learn" call.
  int recent_bits = 1;

  // This is equal to "recent_bits - 1", so has a range from 0 to 254.
  int bit_context = 0;

  // The previous byte. This gets updated after eight bits have been perceived
  // (i.e. recent_bits becomes "1").
  int last_byte = 0;

  unsigned int always_zero = 0;
  unsigned int last_byte_context = 0;
  unsigned int last_two_bytes_context = 0;
  unsigned int last_three_bytes_context = 0;
  unsigned int last_three_bytes_15_bit_hash = 0;
  unsigned int last_four_bytes_context = 0;
  unsigned int last_four_bytes_15_bit_hash = 0;
  unsigned long long last_five_bytes_context = 0;
  unsigned int last_five_bytes_15_bit_hash = 0;
  unsigned int last_five_bytes_21_bit_hash = 0;
  unsigned int second_last_byte = 0;

  // Predictions for the next byte of data from PPM. Each prediction is in the
  // 0-1 range.
  std::valarray<float> ppm_predictions;
  
  // Mixers should call this in their constructor.
  // description: a short identifier for this mixer.
  // ptr: a pointer to this mixer.
  // returns: mixer index.
  int AddMixer(std::string description, Model* ptr) {
    ++num_mixers;
    model_descriptions.push_back(description);
    mixer_index_to_model_ptr[num_mixers - 1] = ptr;
    return num_mixers - 1;
  }
  std::valarray<float> mixer_outputs;
  int num_mixers = 0;
  float final_mixer_output = 0.5;
  std::unordered_map<int, Model*> mixer_index_to_model_ptr;

  const Sigmoid& sigmoid;  // Does not need serialization.

  Nonstationary nonstationary;  // Does not need serialization.

  // Longest match from Match models. Range is 0-7.
  // 0 = 0-3 bytes matched.
  // 1 = 4-7 bytes matched.
  // ...
  // 7 = Over 28 bytes matched.
  unsigned int longest_match = 0;

  // Number of bits seen.
  unsigned long long bits_seen = 0;

  // This is the cross entropy for each model (used for analysis).
  std::valarray<double> entropy;
};

#endif  // SHORT_TERM_MEMORY_H_
