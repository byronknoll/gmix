#include "direct.h"

Direct::Direct(ShortTermMemory& short_term_memory,
               LongTermMemory& long_term_memory, int limit,
               unsigned long long& context)
    : limit_(limit), min_learning_rate_(1.0 / limit), context_(context) {
  prediction_index_ = short_term_memory.num_predictions++;

  memory_index_ = long_term_memory.direct.size();
  long_term_memory.direct.push_back(
      std::unique_ptr<DirectMemory>(new DirectMemory()));
}

void Direct::Predict(ShortTermMemory& short_term_memory,
                     const LongTermMemory& long_term_memory) {
  unsigned int context = (context_ << 8) + short_term_memory.bit_context;
  float p = 0.5;
  const auto& pred_map = long_term_memory.direct[memory_index_]->predictions;
  const auto& it = pred_map.find(context);
  if (it != pred_map.end()) {
    p = it->second.prediction;
  }
  short_term_memory.SetPrediction(p, prediction_index_);
}

void Direct::Learn(const ShortTermMemory& short_term_memory,
                   LongTermMemory& long_term_memory) {
  float learning_rate = min_learning_rate_;
  unsigned int context = (context_ << 8) + short_term_memory.bit_context;
  DirectPrediction& prediction =
      long_term_memory.direct[memory_index_]->predictions[context];
  if (prediction.count < limit_) {
    ++prediction.count;
    learning_rate = 1.0 / prediction.count;
  }
  prediction.prediction +=
      (short_term_memory.new_bit - prediction.prediction) * learning_rate;
}