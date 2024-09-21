#include "predictor.h"

#include <numeric>

#include "contexts/basic-contexts.h"
#include "mixer/mixer.h"
#include "models/direct.h"
#include "models/lstm-model.h"

Predictor::Predictor() : sigmoid_(100001), short_term_memory_(sigmoid_) {
  AddModel(new BasicContexts());
  AddDirect();
  // AddModel(new LstmModel(short_term_memory_, long_term_memory_));
  AddMixers();
  short_term_memory_.predictions.resize(short_term_memory_.num_predictions);
  short_term_memory_.predictions = 0.5;
  short_term_memory_.mixer_outputs.resize(short_term_memory_.num_mixers);
  short_term_memory_.mixer_outputs = 0.5;
}

void Predictor::AddModel(Model* model) {
  models_.push_back(std::unique_ptr<Model>(model));
}

void Predictor::AddDirect() {
  AddModel(new Direct(short_term_memory_, long_term_memory_, 30,
                      short_term_memory_.always_zero, 1));

  AddModel(new Direct(short_term_memory_, long_term_memory_, 30,
                      short_term_memory_.last_byte_context, 256));

  AddModel(new Direct(short_term_memory_, long_term_memory_, 30,
                      short_term_memory_.last_two_bytes_context, 256 * 256));
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
  std::ofstream data_out(path, std::ios::out | std::ios::binary);
  if (!data_out.is_open()) return;

  for (const auto& model : models_) {
    model->WriteToDisk(&data_out);
  }
  short_term_memory_.WriteToDisk(&data_out);
  long_term_memory_.WriteToDisk(&data_out);

  data_out.close();
}

void Predictor::ReadCheckpoint(std::string path) {
  std::ifstream data_in(path, std::ios::in | std::ios::binary);
  if (!data_in.is_open()) return;

  for (const auto& model : models_) {
    model->ReadFromDisk(&data_in);
  }
  short_term_memory_.ReadFromDisk(&data_in);
  long_term_memory_.ReadFromDisk(&data_in);

  data_in.close();
}
