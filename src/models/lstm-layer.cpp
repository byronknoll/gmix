#include "lstm-layer.h"

#include <math.h>

#include <algorithm>
#include <numeric>

#include "../mixer/sigmoid.h"
#ifdef P2_FASTMATH
#include "../mixer/fastmath.h"
#define FAST_TANH fast_tanh
#define FAST_TANH_VEC fast_tanh_vec
#define LSTM_LOGISTIC fast_logistic
#define LSTM_EXPF fast_expf
#else
#define FAST_TANH tanhf
#define FAST_TANH_VEC tanhf
#define LSTM_LOGISTIC Sigmoid::Logistic
#define LSTM_EXPF expf
#endif

namespace {

#ifdef STOCK_VALARRAY
void Adam(std::valarray<float>* g, std::valarray<float>* m,
          std::valarray<float>* v, std::valarray<float>* w, float learning_rate,
          float t, unsigned long long update_limit) {
  if (t < 1.0f) t = 1.0f;
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
#else
void Adam(std::valarray<float>* g, std::valarray<float>* m,
          std::valarray<float>* v, std::valarray<float>* w, float learning_rate,
          float t, unsigned long long update_limit) {
  if (t < 1.0f) t = 1.0f;
  const float beta1 = 0.025f, beta2 = 0.9999f, eps = 1e-6f;
  float t_used = (t < update_limit) ? t : static_cast<float>(update_limit);
  float alpha = learning_rate * 0.1f / std::sqrt(5e-5f * t_used + 1.0f);
  float inv_b1 = 1.0f / (1.0f - std::pow(beta1, t_used));
  float inv_b2 = 1.0f / (1.0f - std::pow(beta2, t_used));
  const size_t sz = g->size();
  float* gp = &(*g)[0];
  float* mp = &(*m)[0];
  float* vp = &(*v)[0];
  float* wp = &(*w)[0];
  for (size_t i = 0; i < sz; ++i) {
    float gi = gp[i];
    float mi = mp[i] * beta1 + (1.0f - beta1) * gi;
    float vi = vp[i] * beta2 + (1.0f - beta2) * gi * gi;
    mp[i] = mi;
    vp[i] = vi;
    float m_hat = mi * inv_b1;
    float v_hat = vi * inv_b2;
    wp[i] -= alpha * (m_hat / (std::sqrt(v_hat + eps)));
  }
}
#endif

}  // namespace

NeuronLayer::NeuronLayer(unsigned int input_size, unsigned int num_cells,
                         int horizon, int offset,
                         LstmMemory& lstm_memory)
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
#ifdef P3_BATCH
  error_hist_.resize(horizon);
  for (int e = 0; e < horizon; ++e) error_hist_[e].resize(num_cells);
#endif
  layer_index_ = lstm_memory.neuron_layer_weights.size();
  lstm_memory.neuron_layer_weights.push_back(
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
                     LstmMemory& lstm_memory)
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
                   lstm_memory),
      input_node_(input_size, num_cells, horizon, output_size_ + input_size_,
                  lstm_memory),
      output_gate_(input_size, num_cells, horizon, output_size_ + input_size_,
                   lstm_memory) {
  float val = sqrt(6.0f / float(input_size_ + output_size_));
  float low = -val;
  float range = 2 * val;
  auto& fg_weights =
      lstm_memory.neuron_layer_weights[forget_gate_.layer_index_].weights;
  auto& in_weights =
      lstm_memory.neuron_layer_weights[input_node_.layer_index_].weights;
  auto& og_weights =
      lstm_memory.neuron_layer_weights[output_gate_.layer_index_].weights;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    for (unsigned int j = 0; j < fg_weights[i].size(); ++j) {
      fg_weights[i][j] = low + Rand() * range;
      in_weights[i][j] = low + Rand() * range;
      og_weights[i][j] = low + Rand() * range;
    }
    fg_weights[i][input_size - 1] = 1;
  }
#ifdef P3_BATCH
  input_hist_.resize(horizon, nullptr);
  sym_hist_.resize(horizon, 0);
#endif
#ifdef LSTM_QUANT
  qcols_ = input_size - output_size - 1;
  qstride_ = (qcols_ + (LSTM_QUANT_PAD - 1)) & ~(LSTM_QUANT_PAD - 1);
  if (qcols_ == 0 || qstride_ > 65536u) {
    fprintf(stderr, "LSTM_QUANT: unsupported width qcols=%u\n", qcols_);
    abort();
  }
  qinput_.assign(qstride_, 0);
#ifdef LSTM_QUANT_KERNEL_VNNI
  qinput_biased_.assign(qstride_, 128);
#endif
  NeuronLayer* gates[3] = {&forget_gate_, &input_node_, &output_gate_};
  for (int g = 0; g < 3; ++g) {
    gates[g]->qweights_.assign((size_t)num_cells_ * qstride_, 0);
    gates[g]->qscale_.assign(num_cells_, 1.0f);
    gates[g]->qrowsum_.assign(num_cells_, 0);
  }
#endif
}

void LstmLayer::ForwardPass(const std::valarray<float>& input, int input_symbol,
                            std::valarray<float>* hidden, int hidden_start,
                            const LstmMemory& lstm_memory) {
  last_state_[epoch_] = state_;
#ifdef LSTM_QUANT
  if (qdirty_) {
    QuantRequantize(lstm_memory);
    qdirty_ = false;
  }
  QuantizeInput(input);
#endif
  ForwardPass(forget_gate_, input, input_symbol, lstm_memory);
  ForwardPass(input_node_, input, input_symbol, lstm_memory);
  ForwardPass(output_gate_, input, input_symbol, lstm_memory);
#ifdef STOCK_VALARRAY
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
#else
  for (unsigned int i = 0; i < num_cells_; ++i) {
    forget_gate_.state_[epoch_][i] =
        LSTM_LOGISTIC(forget_gate_.state_[epoch_][i]);
    input_node_.state_[epoch_][i] = FAST_TANH(input_node_.state_[epoch_][i]);
    output_gate_.state_[epoch_][i] =
        LSTM_LOGISTIC(output_gate_.state_[epoch_][i]);
  }
  input_gate_state_[epoch_] = 1.0f - forget_gate_.state_[epoch_];
  state_ *= forget_gate_.state_[epoch_];
  state_ += input_node_.state_[epoch_] * input_gate_state_[epoch_];
#ifdef P2_FASTMATH
  tanh_state_[epoch_] = FAST_TANH_VEC(state_);
#else
  for (unsigned int i = 0; i < num_cells_; ++i) {
    tanh_state_[epoch_][i] = FAST_TANH(state_[i]);
  }
#endif
  std::slice slice = std::slice(hidden_start, num_cells_, 1);
  (*hidden)[slice] = output_gate_.state_[epoch_] * tanh_state_[epoch_];
#endif
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
}

void LstmLayer::ForwardPass(NeuronLayer& neurons,
                            const std::valarray<float>& input, int input_symbol,
                            const LstmMemory& lstm_memory) {
  const auto& weights =
      lstm_memory.neuron_layer_weights[neurons.layer_index_].weights;
#ifdef STOCK_VALARRAY
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
#else
  float sum_sq = 0.0f;
  float* norm_row = &neurons.norm_[epoch_][0];
#ifdef LSTM_QUANT
  const unsigned int bias_col = (unsigned int)weights[0].size() - 1;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    const LstmQuantAcc acc = QuantDotRow(neurons, i);
    float f = weights[i][input_symbol] + weights[i][bias_col] +
              (float)acc * (neurons.qscale_[i] * qact_scale_);
    norm_row[i] = f;
    sum_sq += f * f;
  }
#else
  const float* inp = &input[0];
  const size_t inp_sz = input.size();
  for (unsigned int i = 0; i < num_cells_; ++i) {
    const float* w_row = &weights[i][output_size_];
    float f = weights[i][input_symbol];
    for (size_t j = 0; j < inp_sz; ++j) {
      f += inp[j] * w_row[j];
    }
    norm_row[i] = f;
    sum_sq += f * f;
  }
#endif
  float ivar = 1.0f / std::sqrt((sum_sq / num_cells_) + 1e-5f);
  neurons.ivar_[epoch_] = ivar;
  float* state_row = &neurons.state_[epoch_][0];
  const float* gamma = &neurons.gamma_[0];
  const float* beta = &neurons.beta_[0];
  for (unsigned int i = 0; i < num_cells_; ++i) {
    norm_row[i] *= ivar;
    state_row[i] = norm_row[i] * gamma[i] + beta[i];
  }
#endif
}

void LstmLayer::ClipGradients(std::valarray<float>* arr) {
#ifdef STOCK_VALARRAY
  for (unsigned int i = 0; i < arr->size(); ++i) {
    if ((*arr)[i] < -gradient_clip_)
      (*arr)[i] = -gradient_clip_;
    else if ((*arr)[i] > gradient_clip_)
      (*arr)[i] = gradient_clip_;
  }
#else
  float* p = &(*arr)[0];
  const size_t sz = arr->size();
  const float gc = gradient_clip_;
  for (size_t i = 0; i < sz; ++i) {
    if (p[i] < -gc)
      p[i] = -gc;
    else if (p[i] > gc)
      p[i] = gc;
  }
#endif
}

void LstmLayer::BackwardPass(const std::valarray<float>& input, int epoch,
                             int layer, int input_symbol,
                             std::valarray<float>* hidden_error,
                             LstmMemory& lstm_memory) {
#ifdef STOCK_VALARRAY
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
#else
  if (epoch == (int)horizon_ - 1) {
    stored_error_ = *hidden_error;
    state_error_ = 0;
  } else {
    stored_error_ += *hidden_error;
  }

  float* out_err = &output_gate_.error_[0];
  float* in_err = &input_node_.error_[0];
  float* f_err = &forget_gate_.error_[0];
  float* st_err = &state_error_[0];
  const float* stored_err = &stored_error_[0];
  const float* tanh_s = &tanh_state_[epoch][0];
  const float* out_s = &output_gate_.state_[epoch][0];
  const float* in_s = &input_node_.state_[epoch][0];
  const float* f_s = &forget_gate_.state_[epoch][0];
  const float* in_gate_s = &input_gate_state_[epoch][0];
  const float* last_s = &last_state_[epoch][0];

  for (unsigned int i = 0; i < num_cells_; ++i) {
    float se = stored_err[i];
    float ts = tanh_s[i];
    float os = out_s[i];
    out_err[i] = ts * se * os * (1.0f - os);
    float s_err = st_err[i] + se * os * (1.0f - ts * ts);
    float is = in_s[i];
    float igs = in_gate_s[i];
    in_err[i] = s_err * igs * (1.0f - is * is);
    float fs = f_s[i];
    f_err[i] = (last_s[i] - is) * s_err * fs * igs;
    st_err[i] = s_err;
  }

  *hidden_error = 0;
  if (epoch > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      st_err[i] *= f_s[i];
    }
    stored_error_ = 0;
  } else {
    if (update_steps_ < update_limit_) {
      ++update_steps_;
    }
  }
#endif

#ifdef P3_BATCH
  input_hist_[epoch] = &input;
  sym_hist_[epoch] = input_symbol;
#endif

  BackwardPass(forget_gate_, input, epoch, layer, input_symbol, hidden_error,
               lstm_memory);
  BackwardPass(input_node_, input, epoch, layer, input_symbol, hidden_error,
               lstm_memory);
  BackwardPass(output_gate_, input, epoch, layer, input_symbol, hidden_error,
               lstm_memory);

#ifdef LSTM_QUANT
  if (epoch == 0) qdirty_ = true;
#endif

  ClipGradients(&state_error_);
  ClipGradients(&stored_error_);
  ClipGradients(hidden_error);
}

void LstmLayer::BackwardPass(NeuronLayer& neurons,
                             const std::valarray<float>& input, int epoch,
                             int layer, int input_symbol,
                             std::valarray<float>* hidden_error,
                             LstmMemory& lstm_memory) {
  auto& weights =
      lstm_memory.neuron_layer_weights[neurons.layer_index_].weights;
#ifdef STOCK_VALARRAY
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
#else
  if (epoch == (int)horizon_ - 1) {
    neurons.gamma_u_ = 0;
    neurons.beta_u_ = 0;
    int offset = output_size_ + input_size_;
    const size_t tr_sz = neurons.transpose_.size();
    for (unsigned int i = 0; i < num_cells_; ++i) {
      neurons.update_[i] = 0;
      const float* w_row = &weights[i][offset];
      for (size_t j = 0; j < tr_sz; ++j) {
        neurons.transpose_[j][i] = w_row[j];
      }
    }
  }
  const float* norm_ep = &neurons.norm_[epoch][0];
  float* err_ptr = &neurons.error_[0];
  float err_norm_sum = 0.0f;
  const float ivar = neurons.ivar_[epoch];
  const float* gamma = &neurons.gamma_[0];
  float* beta_u = &neurons.beta_u_[0];
  float* gamma_u = &neurons.gamma_u_[0];

  for (unsigned int i = 0; i < num_cells_; ++i) {
    float e = err_ptr[i];
    beta_u[i] += e;
    gamma_u[i] += e * norm_ep[i];
    e *= gamma[i] * ivar;
    err_ptr[i] = e;
    err_norm_sum += e * norm_ep[i];
  }
  float mean_err_norm = err_norm_sum / num_cells_;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    err_ptr[i] -= mean_err_norm * norm_ep[i];
  }

  if (layer > 0) {
    float* h_err = &(*hidden_error)[0];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      const float* tr_row = &neurons.transpose_[num_cells_ + i][0];
      float f = 0.0f;
      for (unsigned int j = 0; j < num_cells_; ++j) {
        f += err_ptr[j] * tr_row[j];
      }
      h_err[i] += f;
    }
  }
  if (epoch > 0) {
    float* st_err = &stored_error_[0];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      const float* tr_row = &neurons.transpose_[i][0];
      float f = 0.0f;
      for (unsigned int j = 0; j < num_cells_; ++j) {
        f += err_ptr[j] * tr_row[j];
      }
      st_err[i] += f;
    }
  }
#ifndef P3_BATCH
  const float* inp = &input[0];
  const size_t inp_sz = input.size();
  for (unsigned int i = 0; i < num_cells_; ++i) {
    float e = err_ptr[i];
    float* upd = &neurons.update_[i][output_size_];
    for (size_t j = 0; j < inp_sz; ++j) {
      upd[j] += e * inp[j];
    }
    neurons.update_[i][input_symbol] += e;
  }
#else
  neurons.error_hist_[epoch] = neurons.error_;
  if (epoch == 0) {
#if defined(P3_REGBLOCK)
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float* up = &neurons.update_[i][0];
      float* upd = up + output_size_;
      for (int e = horizon_ - 1; e >= 0; --e)
        up[sym_hist_[e]] += neurons.error_hist_[e][i];
      const unsigned int n = (unsigned int)input_hist_[0]->size();
      constexpr unsigned int BLK = 32u;
      for (unsigned int jb = 0; jb < n; jb += BLK) {
        const unsigned int bn = (jb + BLK <= n) ? BLK : (n - jb);
        float acc[BLK];
        for (unsigned int k = 0; k < bn; ++k) acc[k] = upd[jb + k];
        for (int e = horizon_ - 1; e >= 0; --e) {
          const float fe = neurons.error_hist_[e][i];
          const float* ip = &(*input_hist_[e])[0] + jb;
          for (unsigned int k = 0; k < bn; ++k) acc[k] += fe * ip[k];
        }
        for (unsigned int k = 0; k < bn; ++k) upd[jb + k] = acc[k];
      }
    }
#elif defined(P3_MICROKERNEL) && defined(__AVX512F__)
    const unsigned int H = (unsigned int)horizon_;
    const unsigned int n = (unsigned int)input_hist_[0]->size();
    const float* Xp[512]; const float* Ep[512];
    for (unsigned int e = 0; e < H; ++e) {
      Xp[e] = &(*input_hist_[e])[0];
      Ep[e] = &neurons.error_hist_[e][0];
    }
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float* up = &neurons.update_[i][0];
      for (int e = (int)H - 1; e >= 0; --e) up[sym_hist_[e]] += Ep[e][i];
    }
    constexpr unsigned int MR = 8;
    unsigned int ib = 0;
    for (; ib + MR <= num_cells_; ib += MR) {
      float* upd[MR];
      for (unsigned int c = 0; c < MR; ++c)
        upd[c] = &neurons.update_[ib + c][output_size_];
      unsigned int j = 0;
      for (; j + 16 <= n; j += 16) {
        __m512 acc[MR];
        for (unsigned int c = 0; c < MR; ++c) acc[c] = _mm512_loadu_ps(upd[c] + j);
        for (int e = (int)H - 1; e >= 0; --e) {
          const __m512 x = _mm512_loadu_ps(Xp[e] + j);
          for (unsigned int c = 0; c < MR; ++c)
            acc[c] = _mm512_fmadd_ps(_mm512_set1_ps(Ep[e][ib + c]), x, acc[c]);
        }
        for (unsigned int c = 0; c < MR; ++c) _mm512_storeu_ps(upd[c] + j, acc[c]);
      }
      for (; j < n; ++j)
        for (int e = (int)H - 1; e >= 0; --e) {
          const float xj = Xp[e][j];
          for (unsigned int c = 0; c < MR; ++c) upd[c][j] += Ep[e][ib + c] * xj;
        }
    }
    for (; ib < num_cells_; ++ib) {
      float* upd = &neurons.update_[ib][output_size_];
      for (int e = (int)H - 1; e >= 0; --e) {
        const float fe = Ep[e][ib];
        const float* ip = Xp[e];
        for (unsigned int j = 0; j < n; ++j) upd[j] += fe * ip[j];
      }
    }
#else
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float* up = &neurons.update_[i][0];
      float* upd = up + output_size_;
      for (int e = horizon_ - 1; e >= 0; --e) {
        const float fe = neurons.error_hist_[e][i];
        const float* ip = &(*input_hist_[e])[0];
        const unsigned int n = input_hist_[e]->size();
        for (unsigned int j = 0; j < n; ++j) upd[j] += fe * ip[j];
        up[sym_hist_[e]] += fe;
      }
    }
#endif
  }
#endif
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
#endif
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
#ifdef LSTM_QUANT
  qdirty_ = true;
#endif
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

#ifdef LSTM_QUANT
void LstmLayer::QuantRequantizeGate(NeuronLayer& neurons,
                                    const LstmMemory& lstm_memory) {
  const auto& weights =
      lstm_memory.neuron_layer_weights[neurons.layer_index_].weights;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    neurons.qscale_[i] = LstmQuantizeRow(
        &weights[i][output_size_], qcols_,
        &neurons.qweights_[(size_t)i * qstride_], qstride_,
        &neurons.qrowsum_[i]);
  }
}

void LstmLayer::QuantRequantize(const LstmMemory& lstm_memory) {
  QuantRequantizeGate(forget_gate_, lstm_memory);
  QuantRequantizeGate(input_node_, lstm_memory);
  QuantRequantizeGate(output_gate_, lstm_memory);
}

void LstmLayer::QuantizeInput(const std::valarray<float>& input) {
  if ((unsigned int)input.size() != qcols_ + 1) {
    fprintf(stderr, "LSTM_QUANT: input width %u != qcols+1 %u\n",
            (unsigned int)input.size(), qcols_ + 1);
    abort();
  }
  qact_scale_ = LstmQuantizeActs(&input[0], qcols_, qinput_.data(), qstride_);
#ifdef LSTM_QUANT_KERNEL_VNNI
  for (unsigned int j = 0; j < qstride_; ++j) {
    qinput_biased_[j] = (uint8_t)((int32_t)qinput_[j] + 128);
  }
#endif
}

LstmQuantAcc LstmLayer::QuantDotRow(const NeuronLayer& neurons,
                                    unsigned int i) const {
#ifdef LSTM_QUANT_KERNEL_VNNI
  return LstmQuantDot(&neurons.qweights_[(size_t)i * qstride_], qinput_.data(),
                      qinput_biased_.data(), qstride_, neurons.qrowsum_[i]);
#else
  return LstmQuantDot(&neurons.qweights_[(size_t)i * qstride_], qinput_.data(),
                      nullptr, qstride_, neurons.qrowsum_[i]);
#endif
}
#endif