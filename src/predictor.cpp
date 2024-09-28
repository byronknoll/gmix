#include "predictor.h"

#include <numeric>

#include "contexts/basic-contexts.h"
#include "mixer/mixer.h"
#include "models/direct.h"
#include "models/lstm-model.h"

Predictor::Predictor() : sigmoid_(100001), short_term_memory_(sigmoid_) {
  srand(0xDEADBEEF);
  AddModel(new BasicContexts());
  AddDirect();
  AddModel(new LstmModel(short_term_memory_, long_term_memory_));
  AddMixers();
  short_term_memory_.predictions.resize(short_term_memory_.num_predictions);
  short_term_memory_.predictions = 0.5;
  short_term_memory_.mixer_outputs.resize(short_term_memory_.num_mixers);
  short_term_memory_.mixer_outputs = 0.5;
}

void Predictor::Copy(const Predictor& p) {
  for (int i = 0; i < models_.size(); ++i) {
    models_[i]->Copy(p.models_[i].get());
  }
  long_term_memory_.Copy(&p.long_term_memory_);
  short_term_memory_.Copy(&p.short_term_memory_);
}

void Predictor::AddModel(Model* model) {
  models_.push_back(std::unique_ptr<Model>(model));
}

void Predictor::AddDirect() {
  AddModel(new Direct(short_term_memory_, long_term_memory_, 30,
                      short_term_memory_.always_zero));

  AddModel(new Direct(short_term_memory_, long_term_memory_, 30,
                      short_term_memory_.last_byte_context));

  AddModel(new Direct(short_term_memory_, long_term_memory_, 30,
                      short_term_memory_.last_two_bytes_context));

  AddModel(new Direct(short_term_memory_, long_term_memory_, 30,
                      short_term_memory_.last_three_bytes_16_bit_hash));

  AddModel(new Direct(short_term_memory_, long_term_memory_, 30,
                      short_term_memory_.last_four_bytes_16_bit_hash));
}

void Predictor::AddMixers() {
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_byte_context,
                     short_term_memory_.predictions, 0.005, false));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.always_zero,
                     short_term_memory_.predictions, 0.005, false));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_two_bytes_context,
                     short_term_memory_.predictions, 0.005, false));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_three_bytes_16_bit_hash,
                     short_term_memory_.predictions, 0.005, false));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_four_bytes_16_bit_hash,
                     short_term_memory_.predictions, 0.005, false));

  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.always_zero,
                     short_term_memory_.mixer_outputs, 0.005, true));
}

float Predictor::Predict() {
  for (const auto& model : models_) {
    model->Predict(short_term_memory_, long_term_memory_);
  }
  float prob = Sigmoid::Logistic(short_term_memory_.final_mixer_output);
  float eps = 0.0001;
  if (prob < eps)
    prob = eps;
  else if (prob > 1 - eps)
    prob = 1 - eps;
  return prob;
}

void Predictor::Perceive(int bit) { short_term_memory_.new_bit = bit; }

void Predictor::Learn() {
  for (const auto& model : models_) {
    model->Learn(short_term_memory_, long_term_memory_);
  }
}

void Predictor::WriteCheckpoint(std::string path) {
  std::ofstream data_out_short(path + ".short",
                               std::ios::out | std::ios::binary);
  if (!data_out_short.is_open()) return;
  std::ofstream data_out_long(path + ".long", std::ios::out | std::ios::binary);
  if (!data_out_long.is_open()) return;

  for (const auto& model : models_) {
    model->WriteToDisk(&data_out_short);
  }
  short_term_memory_.WriteToDisk(&data_out_short);
  long_term_memory_.WriteToDisk(&data_out_long);

  data_out_short.close();
  data_out_long.close();
}

void Predictor::ReadCheckpoint(std::string path) {
  std::ifstream data_in_short(path + ".short", std::ios::in | std::ios::binary);
  if (!data_in_short.is_open()) return;
  std::ifstream data_in_long(path + ".long", std::ios::in | std::ios::binary);
  if (!data_in_long.is_open()) return;

  for (const auto& model : models_) {
    model->ReadFromDisk(&data_in_short);
  }
  short_term_memory_.ReadFromDisk(&data_in_short);
  long_term_memory_.ReadFromDisk(&data_in_long);

  data_in_short.close();
  data_in_long.close();
}
