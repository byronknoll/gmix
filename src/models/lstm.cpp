#include "lstm.h"

#include <stdlib.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

#include "../mixer/sigmoid.h"
#ifdef P2_FASTMATH
#include "../mixer/fastmath.h"
#define LSTM_EXPF fast_expf
#else
#define LSTM_EXPF expf
#endif

Lstm::Lstm(unsigned int input_size, unsigned int output_size,
           unsigned int num_cells, unsigned int num_layers, int horizon,
           float learning_rate, float gradient_clip,
           LongTermMemory& long_term_memory)
    : input_history_(horizon),
      hidden_(num_cells * num_layers + 1),
      hidden_error_(num_cells),
      layer_input_(
          std::valarray<std::valarray<float>>(
              std::valarray<float>(input_size + 1 + num_cells * 2), num_layers),
          horizon),
      output_(std::valarray<float>(1.0 / output_size, output_size), horizon),
      learning_rate_(learning_rate),
      num_cells_(num_cells),
      horizon_(horizon),
      input_size_(input_size),
      output_size_(output_size),
      epoch_(0) {
  memory_index_ = long_term_memory.model_memory.size();
  long_term_memory.model_memory.push_back(std::make_unique<LstmMemory>());
  LstmMemory* mem = GetMemory(long_term_memory);
#ifdef P1_MODE
  mem->output_w.resize(output_size);
  for (unsigned int i = 0; i < output_size; ++i) {
    mem->output_w[i].resize(num_cells * num_layers + 1);
  }
#if P1_MODE == 2
  work_.resize(output_size);
  for (unsigned int i = 0; i < output_size; ++i) {
    work_[i].resize(num_cells * num_layers + 1);
  }
#endif
#else
  mem->lstm_output_layer.resize(
      horizon_,
      std::valarray<std::valarray<float>>(
          std::valarray<float>(num_cells * num_layers + 1), output_size));
#endif
  hidden_[hidden_.size() - 1] = 1;
  for (int epoch = 0; epoch < horizon; ++epoch) {
    layer_input_[epoch][0].resize(1 + num_cells + input_size);
    for (unsigned int i = 0; i < num_layers; ++i) {
      layer_input_[epoch][i][layer_input_[epoch][i].size() - 1] = 1;
    }
  }
  for (unsigned int i = 0; i < num_layers; ++i) {
    layers_.push_back(std::unique_ptr<LstmLayer>(new LstmLayer(
        layer_input_[0][i].size() + output_size, input_size_, output_size_,
        num_cells, horizon, gradient_clip, learning_rate, *mem)));
  }
}

LstmMemory* Lstm::GetMemory(LongTermMemory& long_term_memory) {
  return static_cast<LstmMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

const LstmMemory* Lstm::GetMemory(
    const LongTermMemory& long_term_memory) const {
  return static_cast<const LstmMemory*>(
      long_term_memory.model_memory[memory_index_].get());
}

void Lstm::SetInput(const std::valarray<float>& input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    std::copy(begin(input), begin(input) + input_size_,
              begin(layer_input_[epoch_][i]));
  }
}

void Lstm::Perceive(unsigned int input, LongTermMemory& long_term_memory) {
  LstmMemory* mem = GetMemory(long_term_memory);
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  int old_input = input_history_[last_epoch];
  input_history_[last_epoch] = input;
  if (epoch_ == 0) {
#if defined(P1_MODE) && P1_MODE == 2
    for (unsigned int i = 0; i < output_size_; ++i) work_[i] = mem->output_w[i];
#endif
    for (int epoch = horizon_ - 1; epoch >= 0; --epoch) {
      for (int layer = layers_.size() - 1; layer >= 0; --layer) {
        int offset = layer * num_cells_;
#if defined(OERR_REGBLOCK)
        {
          const unsigned int H = (unsigned int)hidden_error_.size();
          const unsigned int OS = output_size_;
          float err[256];
          for (unsigned int i = 0; i < OS; ++i)
            err[i] = (i == (unsigned int)input_history_[epoch])
                         ? (output_[epoch][i] - 1.0f)
                         : output_[epoch][i];
          constexpr unsigned int BLK = 32u;
          for (unsigned int jb = 0; jb < H; jb += BLK) {
            const unsigned int bn = (jb + BLK <= H) ? BLK : (H - jb);
            float acc[BLK];
            for (unsigned int k = 0; k < bn; ++k) acc[k] = hidden_error_[jb + k];
            for (unsigned int i = 0; i < OS; ++i) {
              const float e = err[i];
#if !(defined(P1_MODE) && P1_MODE == 2)
              const float* wr = &mem->lstm_output_layer[epoch][i][jb + offset];
#else
              const float* wr = &work_[i][jb + offset];
#endif
              for (unsigned int k = 0; k < bn; ++k) acc[k] += wr[k] * e;
            }
            for (unsigned int k = 0; k < bn; ++k) hidden_error_[jb + k] = acc[k];
          }
        }
#elif defined(STOCK_VALARRAY)
        for (unsigned int i = 0; i < output_size_; ++i) {
          float error = (i == input_history_[epoch]) ? (output_[epoch][i] - 1)
                                                     : output_[epoch][i];
          for (unsigned int j = 0; j < hidden_error_.size(); ++j) {
            hidden_error_[j] +=
                mem->lstm_output_layer[epoch][i][j + offset] * error;
          }
        }
#else
        for (unsigned int i = 0; i < output_size_; ++i) {
          float error = (i == (unsigned int)input_history_[epoch])
                            ? (output_[epoch][i] - 1.0f)
                            : output_[epoch][i];
#ifndef P1_MODE
          const float* row = &mem->lstm_output_layer[epoch][i][offset];
#elif P1_MODE == 1
          const float* row = &mem->output_w[i][offset];
#else
          const float* row = &work_[i][offset];
#endif
          for (size_t j = 0; j < hidden_error_.size(); ++j) {
            hidden_error_[j] += row[j] * error;
          }
        }
#endif
        int prev_epoch = epoch - 1;
        if (prev_epoch == -1) prev_epoch = horizon_ - 1;
        int input_symbol = input_history_[prev_epoch];
        if (epoch == 0) input_symbol = old_input;
        layers_[layer]->BackwardPass(layer_input_[epoch][layer], epoch, layer,
                                     input_symbol, &hidden_error_,
                                     *mem);
      }
#if defined(P1_MODE) && P1_MODE == 2
      if (epoch > 0) {
        int pe = epoch - 1;
        for (unsigned int i = 0; i < output_size_; ++i) {
          float err = output_[pe][i] -
              ((i == (unsigned int)input_history_[pe]) ? 1.0f : 0.0f);
          float s = learning_rate_ * err;
          for (unsigned int l = 0; l < layers_.size(); ++l) {
            for (unsigned int c = 0; c < num_cells_; ++c) {
              work_[i][l * num_cells_ + c] +=
                  s * layer_input_[epoch][l][input_size_ + c];
            }
          }
        }
      }
#endif
    }
  }

#ifdef STOCK_VALARRAY
  for (unsigned int i = 0; i < output_size_; ++i) {
    float error =
        (i == input) ? (output_[last_epoch][i] - 1) : output_[last_epoch][i];
    mem->lstm_output_layer[epoch_][i] =
        mem->lstm_output_layer[last_epoch][i];
    mem->lstm_output_layer[epoch_][i] -=
        learning_rate_ * error * hidden_;
  }
#else
  for (unsigned int i = 0; i < output_size_; ++i) {
    float error =
        (i == input) ? (output_[last_epoch][i] - 1.0f) : output_[last_epoch][i];
#ifndef P1_MODE
    const float* src = &mem->lstm_output_layer[last_epoch][i][0];
    float* dst = &mem->lstm_output_layer[epoch_][i][0];
    const float scale = learning_rate_ * error;
    const float* hid = &hidden_[0];
    const size_t hid_sz = hidden_.size();
    for (size_t j = 0; j < hid_sz; ++j) {
      dst[j] = src[j] - scale * hid[j];
    }
#else
    const float scale = learning_rate_ * error;
    float* dst = &mem->output_w[i][0];
    const float* hid = &hidden_[0];
    const size_t hid_sz = hidden_.size();
    for (size_t j = 0; j < hid_sz; ++j) {
      dst[j] -= scale * hid[j];
    }
#endif
  }
#endif
}

std::valarray<float>& Lstm::Predict(unsigned int input,
                                    const LongTermMemory& long_term_memory) {
  const LstmMemory* mem = GetMemory(long_term_memory);
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    auto start = begin(hidden_) + i * num_cells_;
    std::copy(start, start + num_cells_,
              begin(layer_input_[epoch_][i]) + input_size_);
    layers_[i]->ForwardPass(layer_input_[epoch_][i], input, &hidden_,
                            i * num_cells_, *mem);
    if (i < layers_.size() - 1) {
      auto start2 =
          begin(layer_input_[epoch_][i + 1]) + num_cells_ + input_size_;
      std::copy(start, start + num_cells_, start2);
    }
  }
#ifdef STOCK_VALARRAY
  float max_out = 0;
  for (unsigned int i = 0; i < output_size_; ++i) {
    float sum = 0;
    for (unsigned int j = 0; j < hidden_.size(); ++j) {
      sum += hidden_[j] * mem->lstm_output_layer[epoch_][i][j];
    }
    output_[epoch_][i] = sum;
    max_out = std::max(sum, max_out);
  }
  for (unsigned int i = 0; i < output_size_; ++i) {
    output_[epoch_][i] = exp(output_[epoch_][i] - max_out);
  }
#else
  const float* hid = &hidden_[0];
  const size_t hid_sz = hidden_.size();
  for (unsigned int i = 0; i < output_size_; ++i) {
#ifndef P1_MODE
    const float* row = &mem->lstm_output_layer[epoch_][i][0];
#else
    const float* row = &mem->output_w[i][0];
#endif
    float sum = 0.0f;
    for (size_t j = 0; j < hid_sz; ++j) {
      sum += hid[j] * row[j];
    }
    output_[epoch_][i] = LSTM_EXPF(sum);
  }
#endif
  output_[epoch_] /= output_[epoch_].sum();
  int epoch = epoch_;
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
  return output_[epoch];
}

void Lstm::WriteToDisk(std::ofstream* s) {
  SerializeArray(s, input_history_);
  SerializeArray(s, hidden_);
  SerializeArray(s, hidden_error_);
  for (auto& x : layer_input_) {
    for (auto& y : x) {
      SerializeArray(s, y);
    }
  }
  for (auto& y : output_) {
    SerializeArray(s, y);
  }
  Serialize(s, epoch_);
  for (auto& layer : layers_) {
    layer->WriteToDisk(s);
  }
}

void Lstm::ReadFromDisk(std::ifstream* s) {
  SerializeArray(s, input_history_);
  SerializeArray(s, hidden_);
  SerializeArray(s, hidden_error_);
  for (auto& x : layer_input_) {
    for (auto& y : x) {
      SerializeArray(s, y);
    }
  }
  for (auto& y : output_) {
    SerializeArray(s, y);
  }
  Serialize(s, epoch_);
  for (auto& layer : layers_) {
    layer->ReadFromDisk(s);
  }
}

void Lstm::Copy(const MemoryInterface* m) {
  const Lstm* orig = static_cast<const Lstm*>(m);
  memory_index_ = orig->memory_index_;
  input_history_ = orig->input_history_;
  hidden_ = orig->hidden_;
  hidden_error_ = orig->hidden_error_;
  layer_input_ = orig->layer_input_;
  output_ = orig->output_;
  epoch_ = orig->epoch_;
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Copy(orig->layers_[i].get());
  }
}

unsigned long long Lstm::GetMemoryUsage() {
  unsigned long long usage = 24;
  usage += 4 * input_history_.size();
  usage += 4 * hidden_.size();
  usage += 4 * hidden_error_.size();
  usage += 4 * layer_input_.size() * layer_input_[0].size() * layer_input_[0][0].size();
  usage += 4 * output_.size() * output_[0].size();
#if defined(P1_MODE) && P1_MODE == 2
  if (work_.size() > 0) {
    usage += 4 * work_.size() * work_[0].size();
  }
#endif
  for (int i = 0; i < layers_.size(); ++i) {
    usage += layers_[i]->GetMemoryUsage();
  }
  return usage;
}

void LstmMemory::WriteToDisk(std::ofstream* s) {
#ifdef P1_MODE
  for (auto& x : output_w) {
    SerializeArray(s, x);
  }
#else
  for (auto& x : lstm_output_layer) {
    for (auto& y : x) {
      SerializeArray(s, y);
    }
  }
#endif
  for (auto& x : neuron_layer_weights) {
    for (auto& y : x.weights) {
      SerializeArray(s, y);
    }
  }
}

void LstmMemory::ReadFromDisk(std::ifstream* s) {
#ifdef P1_MODE
  for (auto& x : output_w) {
    SerializeArray(s, x);
  }
#else
  for (auto& x : lstm_output_layer) {
    for (auto& y : x) {
      SerializeArray(s, y);
    }
  }
#endif
  for (auto& x : neuron_layer_weights) {
    for (auto& y : x.weights) {
      SerializeArray(s, y);
    }
  }
}

void LstmMemory::Copy(const MemoryInterface* m) {
  const LstmMemory* orig = static_cast<const LstmMemory*>(m);
#ifdef P1_MODE
  output_w = orig->output_w;
#else
  lstm_output_layer = orig->lstm_output_layer;
#endif
  neuron_layer_weights.resize(orig->neuron_layer_weights.size(),
                              NeuronLayerWeights(0, 0));
  for (int i = 0; i < neuron_layer_weights.size(); ++i) {
    neuron_layer_weights[i].weights = orig->neuron_layer_weights[i].weights;
  }
}