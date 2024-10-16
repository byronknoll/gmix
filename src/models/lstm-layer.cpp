#include "lstm-layer.h"

#include <math.h>

#include <algorithm>
#include <numeric>

#include "../mixer/sigmoid.h"

namespace {

void Adam(std::valarray<float>* g, std::valarray<float>* m,
          std::valarray<float>* v, std::valarray<float>* w, float learning_rate,
          float t, unsigned long long update_limit) {
  const float beta1 = 0.025, beta2 = 0.9999, eps = 1e-6f;
  float alpha;
  if (t < update_limit) {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f);
  } else {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * update_limit + 1.0f);
  }
  (*m) *= beta1;
  (*m) += (1.0f - beta1) * (*g);
  (*v) *= beta2;
  (*v) += (1.0f - beta2) * (*g) * (*g);
  if (t < update_limit) {
    (*w) -= alpha * (((*m) / (float)(1.0f - pow(beta1, t))) /
                     (sqrt((*v) / (float)(1.0f - pow(beta2, t)) + eps)));
  } else {
    (*w) -=
        alpha * (((*m) / (float)(1.0f - pow(beta1, update_limit))) /
                 (sqrt((*v) / (float)(1.0f - pow(beta2, update_limit)) + eps)));
  }
}

}  // namespace

NeuronLayer::NeuronLayer(unsigned int input_size, unsigned int num_cells,
                         int horizon, int offset,
                         LongTermMemory& long_term_memory)
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
      NeuronLayerWeights(input_size, num_cells));
}

void NeuronLayer::WriteToDisk(std::ofstream* s) {
  SerializeArray(s, error_);
  SerializeArray(s, ivar_);
  SerializeArray(s, gamma_);
  SerializeArray(s, gamma_u_);
  SerializeArray(s, gamma_m_);
  SerializeArray(s, gamma_v_);
  SerializeArray(s, beta_);
  SerializeArray(s, beta_u_);
  SerializeArray(s, beta_m_);
  SerializeArray(s, beta_v_);
  for (auto& x : state_) {
    SerializeArray(s, x);
  }
  for (auto& x : update_) {
    SerializeArray(s, x);
  }
  for (auto& x : m_) {
    SerializeArray(s, x);
  }
  for (auto& x : v_) {
    SerializeArray(s, x);
  }
  for (auto& x : transpose_) {
    SerializeArray(s, x);
  }
  for (auto& x : norm_) {
    SerializeArray(s, x);
  }
}

void NeuronLayer::ReadFromDisk(std::ifstream* s) {
  SerializeArray(s, error_);
  SerializeArray(s, ivar_);
  SerializeArray(s, gamma_);
  SerializeArray(s, gamma_u_);
  SerializeArray(s, gamma_m_);
  SerializeArray(s, gamma_v_);
  SerializeArray(s, beta_);
  SerializeArray(s, beta_u_);
  SerializeArray(s, beta_m_);
  SerializeArray(s, beta_v_);
  for (auto& x : state_) {
    SerializeArray(s, x);
  }
  for (auto& x : update_) {
    SerializeArray(s, x);
  }
  for (auto& x : m_) {
    SerializeArray(s, x);
  }
  for (auto& x : v_) {
    SerializeArray(s, x);
  }
  for (auto& x : transpose_) {
    SerializeArray(s, x);
  }
  for (auto& x : norm_) {
    SerializeArray(s, x);
  }
}

void NeuronLayer::Copy(const MemoryInterface* m) {
  const NeuronLayer* orig = static_cast<const NeuronLayer*>(m);
  error_ = orig->error_;
  ivar_ = orig->ivar_;
  gamma_ = orig->gamma_;
  gamma_u_ = orig->gamma_u_;
  gamma_m_ = orig->gamma_m_;
  gamma_v_ = orig->gamma_v_;
  beta_ = orig->beta_;
  beta_u_ = orig->beta_u_;
  beta_m_ = orig->beta_m_;
  beta_v_ = orig->beta_v_;
  state_ = orig->state_;
  update_ = orig->update_;
  m_ = orig->m_;
  v_ = orig->v_;
  transpose_ = orig->transpose_;
  norm_ = orig->norm_;
}

unsigned long long NeuronLayer::GetMemoryUsage() {
  unsigned long long usage = 4;
  usage += 40 * error_.size();  // 10 valarrays of the same size
  usage += 4 * state_.size() * state_[0].size();
  usage += 4 * update_.size() * update_[0].size();
  usage += 4 * m_.size() * m_[0].size();
  usage += 4 * v_.size() * v_[0].size();
  usage += 4 * transpose_.size() * transpose_[0].size();
  usage += 4 * norm_.size() * norm_[0].size();
  return usage;
}

LstmLayer::LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
                     unsigned int output_size, unsigned int num_cells,
                     int horizon, float gradient_clip, float learning_rate,
                     LongTermMemory& long_term_memory)
    : state_(num_cells),
      state_error_(num_cells),
      stored_error_(num_cells),
      tanh_state_(std::valarray<float>(num_cells), horizon),
      input_gate_state_(std::valarray<float>(num_cells), horizon),
      last_state_(std::valarray<float>(num_cells), horizon),
      gradient_clip_(gradient_clip),
      learning_rate_(learning_rate),
      num_cells_(num_cells),
      horizon_(horizon),
      input_size_(auxiliary_input_size),
      output_size_(output_size),
      epoch_(0),
      forget_gate_(input_size, num_cells, horizon, output_size_ + input_size_,
                   long_term_memory),
      input_node_(input_size, num_cells, horizon, output_size_ + input_size_,
                  long_term_memory),
      output_gate_(input_size, num_cells, horizon, output_size_ + input_size_,
                   long_term_memory) {
  float val = sqrt(6.0f / float(input_size_ + output_size_));
  float low = -val;
  float range = 2 * val;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    for (unsigned int j = 0;
         j < long_term_memory.neuron_layer_weights[0].weights[i].size(); ++j) {
      long_term_memory.neuron_layer_weights[0].weights[i][j] =
          low + Rand() * range;
      long_term_memory.neuron_layer_weights[1].weights[i][j] =
          low + Rand() * range;
      long_term_memory.neuron_layer_weights[2].weights[i][j] =
          low + Rand() * range;
    }

    long_term_memory.neuron_layer_weights[forget_gate_.layer_index_]
        .weights[i][input_size - 1] = 1;
  }
}

void LstmLayer::ForwardPass(const std::valarray<float>& input, int input_symbol,
                            std::valarray<float>* hidden, int hidden_start,
                            const LongTermMemory& long_term_memory) {
  last_state_[epoch_] = state_;
  ForwardPass(forget_gate_, input, input_symbol, long_term_memory);
  ForwardPass(input_node_, input, input_symbol, long_term_memory);
  ForwardPass(output_gate_, input, input_symbol, long_term_memory);
  for (unsigned int i = 0; i < num_cells_; ++i) {
    forget_gate_.state_[epoch_][i] =
        Sigmoid::Logistic(forget_gate_.state_[epoch_][i]);
    input_node_.state_[epoch_][i] = tanh(input_node_.state_[epoch_][i]);
    output_gate_.state_[epoch_][i] =
        Sigmoid::Logistic(output_gate_.state_[epoch_][i]);
  }
  input_gate_state_[epoch_] = 1.0f - forget_gate_.state_[epoch_];
  state_ *= forget_gate_.state_[epoch_];
  state_ += input_node_.state_[epoch_] * input_gate_state_[epoch_];
  tanh_state_[epoch_] = tanh(state_);
  std::slice slice = std::slice(hidden_start, num_cells_, 1);
  (*hidden)[slice] = output_gate_.state_[epoch_] * tanh_state_[epoch_];
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
}

void LstmLayer::ForwardPass(NeuronLayer& neurons,
                            const std::valarray<float>& input, int input_symbol,
                            const LongTermMemory& long_term_memory) {
  const auto& weights =
      long_term_memory.neuron_layer_weights[neurons.layer_index_].weights;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    float f = weights[i][input_symbol];
    for (unsigned int j = 0; j < input.size(); ++j) {
      f += input[j] * weights[i][output_size_ + j];
    }
    neurons.norm_[epoch_][i] = f;
  }
  neurons.ivar_[epoch_] =
      1.0f / sqrt(((neurons.norm_[epoch_] * neurons.norm_[epoch_]).sum() /
                   num_cells_) +
                  1e-5f);
  neurons.norm_[epoch_] *= neurons.ivar_[epoch_];
  neurons.state_[epoch_] =
      neurons.norm_[epoch_] * neurons.gamma_ + neurons.beta_;
}

void LstmLayer::ClipGradients(std::valarray<float>* arr) {
  for (unsigned int i = 0; i < arr->size(); ++i) {
    if ((*arr)[i] < -gradient_clip_)
      (*arr)[i] = -gradient_clip_;
    else if ((*arr)[i] > gradient_clip_)
      (*arr)[i] = gradient_clip_;
  }
}

void LstmLayer::BackwardPass(const std::valarray<float>& input, int epoch,
                             int layer, int input_symbol,
                             std::valarray<float>* hidden_error,
                             LongTermMemory& long_term_memory) {
  if (epoch == (int)horizon_ - 1) {
    stored_error_ = *hidden_error;
    state_error_ = 0;
  } else {
    stored_error_ += *hidden_error;
  }

  output_gate_.error_ = tanh_state_[epoch] * stored_error_ *
                        output_gate_.state_[epoch] *
                        (1.0f - output_gate_.state_[epoch]);
  state_error_ += stored_error_ * output_gate_.state_[epoch] *
                  (1.0f - (tanh_state_[epoch] * tanh_state_[epoch]));
  input_node_.error_ =
      state_error_ * input_gate_state_[epoch] *
      (1.0f - (input_node_.state_[epoch] * input_node_.state_[epoch]));
  forget_gate_.error_ = (last_state_[epoch] - input_node_.state_[epoch]) *
                        state_error_ * forget_gate_.state_[epoch] *
                        input_gate_state_[epoch];

  *hidden_error = 0;
  if (epoch > 0) {
    state_error_ *= forget_gate_.state_[epoch];
    stored_error_ = 0;
  } else {
    if (update_steps_ < update_limit_) {
      ++update_steps_;
    }
  }

  BackwardPass(forget_gate_, input, epoch, layer, input_symbol, hidden_error,
               long_term_memory);
  BackwardPass(input_node_, input, epoch, layer, input_symbol, hidden_error,
               long_term_memory);
  BackwardPass(output_gate_, input, epoch, layer, input_symbol, hidden_error,
               long_term_memory);

  ClipGradients(&state_error_);
  ClipGradients(&stored_error_);
  ClipGradients(hidden_error);
}

void LstmLayer::BackwardPass(NeuronLayer& neurons,
                             const std::valarray<float>& input, int epoch,
                             int layer, int input_symbol,
                             std::valarray<float>* hidden_error,
                             LongTermMemory& long_term_memory) {
  auto& weights =
      long_term_memory.neuron_layer_weights[neurons.layer_index_].weights;
  if (epoch == (int)horizon_ - 1) {
    neurons.gamma_u_ = 0;
    neurons.beta_u_ = 0;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      neurons.update_[i] = 0;
      int offset = output_size_ + input_size_;
      for (unsigned int j = 0; j < neurons.transpose_.size(); ++j) {
        neurons.transpose_[j][i] = weights[i][j + offset];
      }
    }
  }
  neurons.beta_u_ += neurons.error_;
  neurons.gamma_u_ += neurons.error_ * neurons.norm_[epoch];
  neurons.error_ *= neurons.gamma_ * neurons.ivar_[epoch];
  neurons.error_ -=
      ((neurons.error_ * neurons.norm_[epoch]).sum() / num_cells_) *
      neurons.norm_[epoch];
  if (layer > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float f = 0;
      for (unsigned int j = 0; j < num_cells_; ++j) {
        f += neurons.error_[j] * neurons.transpose_[num_cells_ + i][j];
      }
      (*hidden_error)[i] += f;
    }
  }
  if (epoch > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float f = 0;
      for (unsigned int j = 0; j < num_cells_; ++j) {
        f += neurons.error_[j] * neurons.transpose_[i][j];
      }
      stored_error_[i] += f;
    }
  }
  std::slice slice = std::slice(output_size_, input.size(), 1);
  for (unsigned int i = 0; i < num_cells_; ++i) {
    neurons.update_[i][slice] += neurons.error_[i] * input;
    neurons.update_[i][input_symbol] += neurons.error_[i];
  }
  if (epoch == 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      Adam(&neurons.update_[i], &neurons.m_[i], &neurons.v_[i], &weights[i],
           learning_rate_, update_steps_, update_limit_);
    }
    Adam(&neurons.gamma_u_, &neurons.gamma_m_, &neurons.gamma_v_,
         &neurons.gamma_, learning_rate_, update_steps_, update_limit_);
    Adam(&neurons.beta_u_, &neurons.beta_m_, &neurons.beta_v_, &neurons.beta_,
         learning_rate_, update_steps_, update_limit_);
  }
}

void LstmLayer::WriteToDisk(std::ofstream* s) {
  SerializeArray(s, state_);
  SerializeArray(s, state_error_);
  SerializeArray(s, stored_error_);
  for (auto& x : tanh_state_) {
    SerializeArray(s, x);
  }
  for (auto& x : input_gate_state_) {
    SerializeArray(s, x);
  }
  for (auto& x : last_state_) {
    SerializeArray(s, x);
  }
  Serialize(s, epoch_);
  Serialize(s, update_steps_);
  forget_gate_.WriteToDisk(s);
  input_node_.WriteToDisk(s);
  output_gate_.WriteToDisk(s);
}

void LstmLayer::ReadFromDisk(std::ifstream* s) {
  SerializeArray(s, state_);
  SerializeArray(s, state_error_);
  SerializeArray(s, stored_error_);
  for (auto& x : tanh_state_) {
    SerializeArray(s, x);
  }
  for (auto& x : input_gate_state_) {
    SerializeArray(s, x);
  }
  for (auto& x : last_state_) {
    SerializeArray(s, x);
  }
  Serialize(s, epoch_);
  Serialize(s, update_steps_);
  forget_gate_.ReadFromDisk(s);
  input_node_.ReadFromDisk(s);
  output_gate_.ReadFromDisk(s);
}

void LstmLayer::Copy(const MemoryInterface* m) {
  const LstmLayer* orig = static_cast<const LstmLayer*>(m);
  state_ = orig->state_;
  state_error_ = orig->state_error_;
  stored_error_ = orig->stored_error_;
  tanh_state_ = orig->tanh_state_;
  input_gate_state_ = orig->input_gate_state_;
  last_state_ = orig->last_state_;
  epoch_ = orig->epoch_;
  update_steps_ = orig->update_steps_;
  forget_gate_.Copy(&orig->forget_gate_);
  input_node_.Copy(&orig->input_node_);
  output_gate_.Copy(&orig->output_gate_);
}

unsigned long long LstmLayer::GetMemoryUsage() {
  unsigned long long usage = 44;
  usage += 4 * state_.size();
  usage += 4 * state_error_.size();
  usage += 4 * stored_error_.size();
  usage += 4 * tanh_state_.size() * tanh_state_[0].size();
  usage += 4 * input_gate_state_.size() * input_gate_state_[0].size();
  usage += 4 * last_state_.size() * last_state_[0].size();
  usage += forget_gate_.GetMemoryUsage();
  usage += input_node_.GetMemoryUsage();
  usage += output_gate_.GetMemoryUsage();
  return usage;
}