#ifndef MODELS_LSTM_MODEL_H_
#define MODELS_LSTM_MODEL_H_

#include "../model.h"
#include "lstm.h"

class LstmModel : public Model {
 public:
  LstmModel(ShortTermMemory& short_term_memory,
            LongTermMemory& long_term_memory);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory);
  void WriteToDisk(std::ofstream* os) {}
  void ReadFromDisk(std::ifstream* is) {}

 private:
  Lstm lstm_;
  int top_, mid_, bot_, prediction_index_;
  std::valarray<float> probs_;
  bool learning_enabled_ = true;
};

#endif  // MODELS_LSTM_MODEL_H_