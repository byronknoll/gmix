#include "lstm.h"

#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <numeric>

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
  long_term_memory.lstm_output_layer.resize(
      horizon_,
      std::valarray<std::valarray<float>>(
          std::valarray<float>(num_cells * num_layers + 1), output_size));
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
        num_cells, horizon, gradient_clip, learning_rate, long_term_memory)));
  }
}

void Lstm::SetInput(const std::valarray<float>& input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    std::copy(begin(input), begin(input) + input_size_,
              begin(layer_input_[epoch_][i]));
  }
}

void Lstm::Perceive(unsigned int input, LongTermMemory& long_term_memory) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  int old_input = input_history_[last_epoch];
  input_history_[last_epoch] = input;
  if (epoch_ == 0) {
    for (int epoch = horizon_ - 1; epoch >= 0; --epoch) {
      for (int layer = layers_.size() - 1; layer >= 0; --layer) {
        int offset = layer * num_cells_;
        for (unsigned int i = 0; i < output_size_; ++i) {
          float error = (i == input_history_[epoch]) ? (output_[epoch][i] - 1)
                                                     : output_[epoch][i];
          for (unsigned int j = 0; j < hidden_error_.size(); ++j) {
            hidden_error_[j] +=
                long_term_memory.lstm_output_layer[epoch][i][j + offset] *
                error;
          }
        }
        int prev_epoch = epoch - 1;
        if (prev_epoch == -1) prev_epoch = horizon_ - 1;
        int input_symbol = input_history_[prev_epoch];
        if (epoch == 0) input_symbol = old_input;
        layers_[layer]->BackwardPass(layer_input_[epoch][layer], epoch, layer,
                                     input_symbol, &hidden_error_,
                                     long_term_memory);
      }
    }
  }

  for (unsigned int i = 0; i < output_size_; ++i) {
    float error =
        (i == input) ? (output_[last_epoch][i] - 1) : output_[last_epoch][i];
    long_term_memory.lstm_output_layer[epoch_][i] =
        long_term_memory.lstm_output_layer[last_epoch][i];
    long_term_memory.lstm_output_layer[epoch_][i] -=
        learning_rate_ * error * hidden_;
  }
}

std::valarray<float>& Lstm::Predict(unsigned int input,
                                    const LongTermMemory& long_term_memory) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    auto start = begin(hidden_) + i * num_cells_;
    std::copy(start, start + num_cells_,
              begin(layer_input_[epoch_][i]) + input_size_);
    layers_[i]->ForwardPass(layer_input_[epoch_][i], input, &hidden_,
                            i * num_cells_, long_term_memory);
    if (i < layers_.size() - 1) {
      auto start2 =
          begin(layer_input_[epoch_][i + 1]) + num_cells_ + input_size_;
      std::copy(start, start + num_cells_, start2);
    }
  }
  float max_out = 0;
  for (unsigned int i = 0; i < output_size_; ++i) {
    float sum = 0;
    for (unsigned int j = 0; j < hidden_.size(); ++j) {
      sum += hidden_[j] * long_term_memory.lstm_output_layer[epoch_][i][j];
    }
    output_[epoch_][i] = sum;
    max_out = std::max(sum, max_out);
  }
  for (unsigned int i = 0; i < output_size_; ++i) {
    output_[epoch_][i] = exp(output_[epoch_][i] - max_out);
  }
  output_[epoch_] /= output_[epoch_].sum();
  int epoch = epoch_;
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
  return output_[epoch];
}

void Lstm::WriteToDisk(std::ofstream* os) {
  for (unsigned int& i : input_history_) {
    os->write(reinterpret_cast<char*>(&i), sizeof(i));
  }
  for (float& f : hidden_) {
    os->write(reinterpret_cast<char*>(&f), sizeof(f));
  }
  for (float& f : hidden_error_) {
    os->write(reinterpret_cast<char*>(&f), sizeof(f));
  }
  for (auto& x : layer_input_) {
    for (auto& y : x) {
      for (float& z : y) {
        os->write(reinterpret_cast<char*>(&z), sizeof(z));
      }
    }
  }
  for (auto& y : output_) {
    for (float& z : y) {
      os->write(reinterpret_cast<char*>(&z), sizeof(z));
    }
  }
  os->write(reinterpret_cast<char*>(&epoch_), sizeof(epoch_));
  for (auto& layer : layers_) {
    layer->WriteToDisk(os);
  }
}

void Lstm::ReadFromDisk(std::ifstream* is) {
  for (unsigned int& i : input_history_) {
    is->read(reinterpret_cast<char*>(&i), sizeof(i));
  }
  for (float& f : hidden_) {
    is->read(reinterpret_cast<char*>(&f), sizeof(f));
  }
  for (float& f : hidden_error_) {
    is->read(reinterpret_cast<char*>(&f), sizeof(f));
  }
  for (auto& x : layer_input_) {
    for (auto& y : x) {
      for (float& z : y) {
        is->read(reinterpret_cast<char*>(&z), sizeof(z));
      }
    }
  }
  for (auto& y : output_) {
    for (float& z : y) {
      is->read(reinterpret_cast<char*>(&z), sizeof(z));
    }
  }
  is->read(reinterpret_cast<char*>(&epoch_), sizeof(epoch_));
  for (auto& layer : layers_) {
    layer->ReadFromDisk(is);
  }
}