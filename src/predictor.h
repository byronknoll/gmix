#ifndef PREDICTOR_H_
#define PREDICTOR_H_

#include <vector>
#include <memory>

#include "long-term-memory.h"
#include "model.h"
#include "short-term-memory.h"

class Predictor {
 public:
  Predictor();
  float Predict();
  void Perceive(int bit);
  void Learn();

 private:
  LongTermMemory long_term_memory_;
  ShortTermMemory short_term_memory_;
  std::vector<std::unique_ptr<Model>> models_;
};

#endif  // PREDICTOR_H_
