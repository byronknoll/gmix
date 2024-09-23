#ifndef MODELS_LSTM_LAYER_H
#define MODELS_LSTM_LAYER_H

#include <math.h>
#include <stdlib.h>

#include <valarray>
#include <vector>

#include "../memory/long-term-memory.h"

struct NeuronLayer {
  NeuronLayer(unsigned int input_size, unsigned int num_cells, int horizon,
              int offset, LongTermMemory& long_term_memory)
      : error_(num_cells),
        ivar_(horizon),
        gamma_(1.0, num_cells),
        gamma_u_(num_cells),
        gamma_m_(num_cells),
        gamma_v_(num_cells),
        beta_(num_cells),
        beta_u_(num_cells),
        beta_m_(num_cells),
        beta_v_(num_cells),
        state_(std::valarray<float>(num_cells), horizon),
        update_(std::valarray<float>(input_size), num_cells),
        m_(std::valarray<float>(input_size), num_cells),
        v_(std::valarray<float>(input_size), num_cells),
        transpose_(std::valarray<float>(num_cells), input_size - offset),
        norm_(std::valarray<float>(num_cells), horizon) {
    layer_index_ = long_term_memory.neuron_layer_weights.size();
    long_term_memory.neuron_layer_weights.push_back(
        std::unique_ptr<NeuronLayerWeights>(
            new NeuronLayerWeights(input_size, num_cells)));
  };

  std::valarray<float> error_, ivar_, gamma_, gamma_u_, gamma_m_, gamma_v_,
      beta_, beta_u_, beta_m_, beta_v_;
  std::valarray<std::valarray<float>> state_, update_, m_, v_, transpose_,
      norm_;
  int layer_index_;
};

class LstmLayer {
 public:
  LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
            unsigned int output_size, unsigned int num_cells, int horizon,
            float gradient_clip, float learning_rate,
            LongTermMemory& long_term_memory);
  void ForwardPass(const std::valarray<float>& input, int input_symbol,
                   std::valarray<float>* hidden, int hidden_start,
                   const LongTermMemory& long_term_memory);
  void BackwardPass(const std::valarray<float>& input, int epoch, int layer,
                    int input_symbol, std::valarray<float>* hidden_error,
                    LongTermMemory& long_term_memory);
  static inline float Rand() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

 private:
  std::valarray<float> state_, state_error_, stored_error_;
  std::valarray<std::valarray<float>> tanh_state_, input_gate_state_,
      last_state_;
  float gradient_clip_, learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  unsigned long long update_steps_ = 0, update_limit_ = 3000;
  NeuronLayer forget_gate_, input_node_, output_gate_;

  void ClipGradients(std::valarray<float>* arr);
  void ForwardPass(NeuronLayer& neurons, const std::valarray<float>& input,
                   int input_symbol, const LongTermMemory& long_term_memory);
  void BackwardPass(NeuronLayer& neurons, const std::valarray<float>& input,
                    int epoch, int layer, int input_symbol,
                    std::valarray<float>* hidden_error,
                    LongTermMemory& long_term_memory);
};

#endif  // MODELS_LSTM_LAYER_H