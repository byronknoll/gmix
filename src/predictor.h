#ifndef PREDICTOR_H_
#define PREDICTOR_H_

#include <memory>
#include <string>
#include <vector>

#include "memory/long-term-memory.h"
#include "memory/short-term-memory.h"
#include "mixer/sigmoid.h"
#include "model.h"

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
  void WriteCheckpoint(std::string path);
  void ReadCheckpoint(std::string path);
  // Makes this predictor a deep copy of the other Predictor.
  void Copy(const Predictor& p);
  // This will output files (entropy.tsv and memory.tsv) with model metrics.
  // Every "sample_frequency" bits, a new entry will be output.
  void EnableAnalysis(int sample_frequency);
  // This changes the analysis frequency. Every "sample_frequency" bits, a
  // new entry will be output. Set to zero to disable analysis. This should
  // only be called after "EnableAnalysis" is run.
  void SetAnalysisFrequency(int sample_frequency) {
    sample_frequency_ = sample_frequency;
  }

 private:
  LongTermMemory long_term_memory_;
  ShortTermMemory short_term_memory_;
  std::vector<std::unique_ptr<Model>> models_;
  // If sample_frequency_ is >0, model analysis will be enabled.
  int sample_frequency_ = 0;

  void AddIndirect();
  void AddSkip();
  void AddMatch();
  void AddDoubleIndirect();
  void AddMixers();
  void AddIntervalContexts();
  void AddModel(Model* model);
  void RunAnalysis(int bit);
  void UpdateEntropy(int bit, int index);
};

#endif  // PREDICTOR_H_
