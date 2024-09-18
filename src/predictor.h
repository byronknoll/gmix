#ifndef PREDICTOR_H_
#define PREDICTOR_H_

#include <memory>
#include <vector>

#include "long-term-memory.h"
#include "model.h"
#include "short-term-memory.h"
#include "mixer/sigmoid.h"

// This is the main predictor which runs all models to produce a final
// prediction. The Predict+Perceive+Learn functions are similar to the Model
// interface (see model.h).
class Predictor : MemoryInterface {
 public:
  Predictor();
  float Predict();
  void Perceive(int bit);
  void Learn();
  void WriteToDisk() {}
  void ReadFromDisk() {}

 private:
  LongTermMemory long_term_memory_;
  ShortTermMemory short_term_memory_;
  std::vector<std::unique_ptr<Model>> models_;
  Sigmoid sigmoid_;

  void AddDirect();
};

#endif  // PREDICTOR_H_
