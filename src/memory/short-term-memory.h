#ifndef SHORT_TERM_MEMORY_H_
#define SHORT_TERM_MEMORY_H_

#include <valarray>

#include "../memory-interface.h"
#include "../mixer/sigmoid.h"
#include "../contexts/nonstationary.h"

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

  // Predictions for the next bit of data. Each prediction should be a
  // probability between 0 to 1.
  void SetPrediction(float prediction, int index) {
    predictions[index] = sigmoid.Logit(prediction);
  }
  std::valarray<float> predictions;
  int num_predictions = 0;

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

  unsigned long long always_zero = 0;
  unsigned long long last_byte_context = 0;
  unsigned long long last_two_bytes_context = 0;
  unsigned long long last_three_bytes_context = 0;
  unsigned long long last_three_bytes_15_bit_hash = 0;
  unsigned long long last_four_bytes_context = 0;
  unsigned long long last_four_bytes_15_bit_hash = 0;
  unsigned long long last_five_bytes_context = 0;
  unsigned long long last_five_bytes_15_bit_hash = 0;

  // Predictions for the next byte of data from PPM. Each prediction is in the
  // 0-1 range.
  std::valarray<float> ppm_predictions;

  std::valarray<float> mixer_outputs;
  int num_mixers = 0;
  float final_mixer_output = 0.5;

  const Sigmoid& sigmoid;  // Does not need serlialization.

  Nonstationary nonstationary;  // Does not need serlialization.
};

#endif  // SHORT_TERM_MEMORY_H_
