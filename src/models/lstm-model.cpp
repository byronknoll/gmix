#include "lstm-model.h"

#include <numeric>

LstmModel::LstmModel(ShortTermMemory& short_term_memory,
                     LongTermMemory& long_term_memory)
    : lstm_(0, 256, 20, 1, 100, 0.03, 10, long_term_memory),
      top_(255),
      mid_(127),
      bot_(0),
      probs_(1.0 / 256, 256) {
  prediction_index_ = short_term_memory.num_predictions++;
}

void LstmModel::Predict(ShortTermMemory& short_term_memory,
                        const LongTermMemory& long_term_memory) {
  if (short_term_memory.recent_bits == 1) {
    // A new byte has been observed. Update the byte-level predictions.
    probs_ = lstm_.Predict(short_term_memory.last_byte, long_term_memory);
    top_ = 255;
    bot_ = 0;
  } else {
    if (short_term_memory.new_bit) {
      bot_ = mid_ + 1;
    } else {
      top_ = mid_;
    }
  }
  mid_ = bot_ + ((top_ - bot_) / 2);
  float num = std::accumulate(&probs_[mid_ + 1], &probs_[top_ + 1], 0.0f);
  float denom = std::accumulate(&probs_[bot_], &probs_[mid_ + 1], num);
  float p = 0.5;
  if (denom != 0) p = num / denom;
  short_term_memory.SetPrediction(p, prediction_index_);
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