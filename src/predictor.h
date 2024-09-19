#ifndef PREDICTOR_H_
#define PREDICTOR_H_

#include <memory>
#include <string>
#include <vector>

#include "long-term-memory.h"
#include "mixer/sigmoid.h"
#include "model.h"
#include "short-term-memory.h"

// This is the main predictor which runs all models to produce a final
// prediction. The following functions should be called in order:
// (1) Predict: predicts the next bit.
// (2) Perceive: pass in the next bit of input.
// (3) Learn: this updates long term memory to learn from the input. After
// training is finished, calling this is optional since predictions can be made
// using just Predict+Perceive.
class Predictor {
 public:
  Predictor();
  float Predict();
  void Perceive(int bit);
  void Learn();
  void WriteCheckpoint(std::string path) {}
  void ReadCheckpoint(std::string path) {}

 private:
  Sigmoid sigmoid_;
  LongTermMemory long_term_memory_;
  ShortTermMemory short_term_memory_;
  std::vector<std::unique_ptr<Model>> models_;

  void AddDirect();
  void AddMixers();
};

#endif  // PREDICTOR_H_
