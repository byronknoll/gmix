#include "predictor.h"

#include "contexts/basic-contexts.h"
#include "contexts/context-hash.h"
#include "models/direct.h"

Predictor::Predictor() {
  BasicContexts* basic = new BasicContexts();
  models_.push_back(std::unique_ptr<Model>(basic));
  AddDirect();
}

void Predictor::AddDirect() {
  ContextHash* context1 = new ContextHash(1, 8, short_term_memory_.hash_1_8);
  models_.push_back(std::unique_ptr<Model>(context1));
  Direct* direct1 =
      new Direct(short_term_memory_, long_term_memory_, 30, 0,
                 short_term_memory_.hash_1_8, long_term_memory_.direct_1_8);
  models_.push_back(std::unique_ptr<Model>(direct1));

  ContextHash* context2 = new ContextHash(2, 8, short_term_memory_.hash_2_8);
  models_.push_back(std::unique_ptr<Model>(context2));
  Direct* direct2 =
      new Direct(short_term_memory_, long_term_memory_, 30, 0,
                 short_term_memory_.hash_2_8, long_term_memory_.direct_2_8);
  models_.push_back(std::unique_ptr<Model>(direct2));
}

float Predictor::Predict() {
  for (const auto& model : models_) {
    model->Predict(short_term_memory_, long_term_memory_);
  }
  return short_term_memory_.predictions[0];
}

void Predictor::Perceive(int bit) {
  short_term_memory_.new_bit = bit;
  for (const auto& model : models_) {
    model->Perceive(short_term_memory_, long_term_memory_);
  }
}

void Predictor::Learn() {
  for (const auto& model : models_) {
    model->Learn(short_term_memory_, long_term_memory_);
  }
}
