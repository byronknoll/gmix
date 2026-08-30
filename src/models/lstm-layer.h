#ifndef MODELS_LSTM_LAYER_H
#define MODELS_LSTM_LAYER_H

#include <math.h>
#include <stdlib.h>

#include <valarray>
#include <vector>

#include "../memory-interface.h"

struct NeuronLayerWeights {
  NeuronLayerWeights(unsigned int input_size, unsigned int num_cells)
      : weights(std::valarray<float>(input_size), num_cells) {};
  std::valarray<std::valarray<float>> weights;
};

struct LstmMemory : public MemoryInterface {
  std::vector<NeuronLayerWeights> neuron_layer_weights;
  std::valarray<std::valarray<std::valarray<float>>> lstm_output_layer;

  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;
};

struct NeuronLayer : public MemoryInterface {
  NeuronLayer(unsigned int input_size, unsigned int num_cells, int horizon,
              int offset, LstmMemory& lstm_memory);
  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;
  unsigned long long GetMemoryUsage();

  std::valarray<float> error_, ivar_, gamma_, gamma_u_, gamma_m_, gamma_v_,
      beta_, beta_u_, beta_m_, beta_v_;
  std::valarray<std::valarray<float>> state_, update_, m_, v_, transpose_,
      norm_;
  int layer_index_;
};

class LstmLayer : public MemoryInterface {
 public:
  LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
            unsigned int output_size, unsigned int num_cells, int horizon,
            float gradient_clip, float learning_rate,
            LstmMemory& lstm_memory);
  void ForwardPass(const std::valarray<float>& input, int input_symbol,
                   std::valarray<float>* hidden, int hidden_start,
                   const LstmMemory& lstm_memory);
  void BackwardPass(const std::valarray<float>& input, int epoch, int layer,
                    int input_symbol, std::valarray<float>* hidden_error,
                    LstmMemory& lstm_memory);
  static inline float Rand() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;
  unsigned long long GetMemoryUsage();

 private:
  std::valarray<float> state_, state_error_, stored_error_;
  std::valarray<std::valarray<float>> tanh_state_, input_gate_state_,
      last_state_;
  const float gradient_clip_, learning_rate_;
  const unsigned int num_cells_, horizon_, input_size_, output_size_;
  unsigned int epoch_;
  unsigned long long update_steps_ = 0;
  const unsigned long long update_limit_ = 3000;
  NeuronLayer forget_gate_, input_node_, output_gate_;

  void ClipGradients(std::valarray<float>* arr);
  void ForwardPass(NeuronLayer& neurons, const std::valarray<float>& input,
                   int input_symbol, const LstmMemory& lstm_memory);
  void BackwardPass(NeuronLayer& neurons, const std::valarray<float>& input,
                    int epoch, int layer, int input_symbol,
                    std::valarray<float>* hidden_error,
                    LstmMemory& lstm_memory);
};

#endif  // MODELS_LSTM_LAYER_H