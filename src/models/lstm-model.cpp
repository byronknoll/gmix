#include "lstm-model.h"

#include <numeric>

LstmModel::LstmModel(ShortTermMemory& short_term_memory,
                     LongTermMemory& long_term_memory, bool enable_analysis)
    : lstm_(256, 256, 200, 2, 100, 0.03, 10, long_term_memory),
      top_(255),
      mid_(127),
      bot_(0),
      probs_(1.0 / 256, 256) {
  prediction_index_ =
      short_term_memory.AddPrediction("LSTM", enable_analysis, this);
  short_term_memory.models_with_skip_connection.push_back(prediction_index_);
}

void LstmModel::Predict(ShortTermMemory& short_term_memory,
                        const LongTermMemory& long_term_memory) {
  if (short_term_memory.recent_bits == 1) {
    // A new byte has been observed. Update the byte-level predictions.
    lstm_.SetInput(short_term_memory.ppm_predictions);
    probs_ = lstm_.Predict(short_term_memory.last_byte, long_term_memory);
    top_ = 255;
    bot_ = 0;
    // Update the "lstm_prediction" context.
    float max_pred = 0;
    short_term_memory.lstm_prediction_context = 0;
    for (int i = 0; i < 256; ++i) {
      if (probs_[i] > max_pred) {
        max_pred = probs_[i];
        short_term_memory.lstm_prediction_context = i;
      }
    }
    short_term_memory.agreement_context =
        (short_term_memory.lstm_prediction_context ==
         short_term_memory.ppm_prediction_context)
            ? short_term_memory.lstm_prediction_context
            : 256;
    prefix_sum_[0] = 0.0f;
    for (int i = 0; i < 256; ++i) {
      prefix_sum_[i + 1] = prefix_sum_[i] + probs_[i];
    }
  } else {
    if (short_term_memory.new_bit) {
      bot_ = mid_ + 1;
    } else {
      top_ = mid_;
    }
  }
  mid_ = bot_ + ((top_ - bot_) / 2);
  float num = prefix_sum_[top_ + 1] - prefix_sum_[mid_ + 1];
  float denom = prefix_sum_[top_ + 1] - prefix_sum_[bot_];
  if (denom != 0) {
    float p = num / denom;
    short_term_memory.SetPrediction(p, prediction_index_);
    int expected_bit = (p >= 0.5f) ? 1 : 0;
    short_term_memory.lstm_bit_context =
        (expected_bit << 8) | short_term_memory.bit_context;
    int ppm_bit = (short_term_memory.ppm_bit_context >> 8) & 1;
    short_term_memory.bit_agreement_context =
        (ppm_bit << 9) | (expected_bit << 8) | (short_term_memory.bit_context & 0xff);
  }
}

void LstmModel::Learn(const ShortTermMemory& short_term_memory,
                      LongTermMemory& long_term_memory) {
  int current_byte =
      short_term_memory.recent_bits * 2 + short_term_memory.new_bit;
  if (current_byte >= 256) {
    // A new byte has been observed.
    current_byte -= 256;
    lstm_.Perceive(current_byte, long_term_memory);
  }
}

void LstmModel::WriteToDisk(std::ofstream* s) {
  Serialize(s, top_);
  Serialize(s, mid_);
  Serialize(s, bot_);
  SerializeArray(s, probs_);
  lstm_.WriteToDisk(s);
}

void LstmModel::ReadFromDisk(std::ifstream* s) {
  Serialize(s, top_);
  Serialize(s, mid_);
  Serialize(s, bot_);
  SerializeArray(s, probs_);
  lstm_.ReadFromDisk(s);
  prefix_sum_[0] = 0.0f;
  for (int i = 0; i < 256; ++i) {
    prefix_sum_[i + 1] = prefix_sum_[i] + probs_[i];
  }
}

void LstmModel::Copy(const MemoryInterface* m) {
  const LstmModel* orig = static_cast<const LstmModel*>(m);
  top_ = orig->top_;
  mid_ = orig->mid_;
  bot_ = orig->bot_;
  probs_ = orig->probs_;
  prefix_sum_ = orig->prefix_sum_;
  lstm_.Copy(&orig->lstm_);
}

unsigned long long LstmModel::GetMemoryUsage(
    const ShortTermMemory& short_term_memory,
    const LongTermMemory& long_term_memory) {
  unsigned long long usage = 16;
  usage += 256 * 4;  // probs_
  const LstmMemory* mem = lstm_.GetMemory(long_term_memory);
#ifdef P1_MODE
  if (mem->output_w.size() > 0) {
    usage += 4 * mem->output_w.size() * mem->output_w[0].size();
  }
#else
  if (mem->lstm_output_layer.size() > 0) {
    usage += 4 * mem->lstm_output_layer.size() *
             mem->lstm_output_layer[0].size() *
             mem->lstm_output_layer[0][0].size();
  }
#endif
  for (const auto& layer : mem->neuron_layer_weights) {
    usage += 4 * layer.weights.size() * layer.weights[0].size();
  }
  usage += lstm_.GetMemoryUsage();
  return usage;
}