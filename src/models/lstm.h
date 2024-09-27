#ifndef MODELS_LSTM_H
#define MODELS_LSTM_H

#include <memory>
#include <string>
#include <valarray>
#include <vector>

#include "../memory/long-term-memory.h"
#include "../memory-interface.h"
#include "lstm-layer.h"

class Lstm : public MemoryInterface {
 public:
  Lstm(unsigned int input_size, unsigned int output_size,
       unsigned int num_cells, unsigned int num_layers, int horizon,
       float learning_rate, float gradient_clip,
       LongTermMemory& long_term_memory);
  void Perceive(unsigned int input, LongTermMemory& long_term_memory);
  std::valarray<float>& Predict(unsigned int input,
                                const LongTermMemory& long_term_memory);
  void SetInput(const std::valarray<float>& input);
  void WriteToDisk(std::ofstream* s);
  void ReadFromDisk(std::ifstream* s);
  void Copy(const MemoryInterface* m);

 private:
  std::vector<std::unique_ptr<LstmLayer>> layers_;
  std::vector<unsigned int> input_history_;
  std::valarray<float> hidden_, hidden_error_;
  std::valarray<std::valarray<std::valarray<float>>> layer_input_;
  std::valarray<std::valarray<float>> output_;
  const float learning_rate_;
  const unsigned int num_cells_, horizon_, input_size_, output_size_;
  unsigned int epoch_;
};

#endif  // MODELS_LSTM_H