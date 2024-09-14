#include "predictor.h"

#include "contexts/context-hash.h"

Predictor::Predictor() {
  ContextHash* context =
      new ContextHash(1, 8, short_term_memory_.context_hash.hash_1_8);
  models_.push_back(std::unique_ptr<Model>(context));
}

float Predictor::Predict() {
  return 0.5;
}

void Predictor::Perceive(int bit) {

}

void Predictor::Train() {

}
