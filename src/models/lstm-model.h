#ifndef MODELS_LSTM_MODEL_H_
#define MODELS_LSTM_MODEL_H_

#include "../model.h"
#include "lstm.h"

// LSTM predictions are made once per byte, predicting the next byte. The
// byte-level predictions are converted into bit-level predictions.
class LstmModel : public Model {
 public:
  LstmModel(ShortTermMemory& short_term_memory,
            LongTermMemory& long_term_memory, bool enable_analysis);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory);
  void WriteToDisk(std::ofstream* s);
  void ReadFromDisk(std::ifstream* s);
  void Copy(const MemoryInterface* m);
  unsigned long long GetMemoryUsage(const ShortTermMemory& short_term_memory,
                                    const LongTermMemory& long_term_memory);

 private:
  Lstm lstm_;
  // top_, mid_, and bot_ are used to keep track of ranges for converting
  // byte-level predictions to bit-level predictions. The range is updated as
  // bits are observed.
  int top_, mid_, bot_, prediction_index_;
  // This contains 256 entries, with a probability distribution for the next
  // byte prediction.
  std::valarray<float> probs_;
};

#endif  // MODELS_LSTM_MODEL_H_