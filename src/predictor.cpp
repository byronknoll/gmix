#include "predictor.h"

#include <numeric>

#include "contexts/basic-contexts.h"
#include "mixer/mixer.h"
#include "models/direct.h"

Predictor::Predictor() : sigmoid_(100001), short_term_memory_(sigmoid_) {
  BasicContexts* basic = new BasicContexts();
  models_.push_back(std::unique_ptr<Model>(basic));
  AddDirect();
  AddMixers();
  short_term_memory_.predictions.resize(short_term_memory_.num_predictions);
  short_term_memory_.predictions = 0.5;
  short_term_memory_.mixer_outputs.resize(short_term_memory_.num_mixers);
  short_term_memory_.mixer_outputs = 0.5;
}

void Predictor::AddDirect() {
  Direct* direct0 = new Direct(short_term_memory_, long_term_memory_, 30,
                               short_term_memory_.always_zero, 1);
  models_.push_back(std::unique_ptr<Model>(direct0));

  Direct* direct1 = new Direct(short_term_memory_, long_term_memory_, 30,
                               short_term_memory_.last_byte_context, 256);
  models_.push_back(std::unique_ptr<Model>(direct1));

  Direct* direct2 =
      new Direct(short_term_memory_, long_term_memory_, 30,
                 short_term_memory_.last_two_bytes_context, 256 * 256);
  models_.push_back(std::unique_ptr<Model>(direct2));
}

void Predictor::AddMixers() {
  Mixer* mixer1 = new Mixer(short_term_memory_, long_term_memory_,
                            short_term_memory_.last_byte_context,
                            short_term_memory_.predictions, 0.005, false);
  models_.push_back(std::unique_ptr<Model>(mixer1));
  Mixer* mixer2 = new Mixer(short_term_memory_, long_term_memory_,
                            short_term_memory_.always_zero,
                            short_term_memory_.predictions, 0.005, false);
  models_.push_back(std::unique_ptr<Model>(mixer2));
  Mixer* mixer3 = new Mixer(short_term_memory_, long_term_memory_,
                            short_term_memory_.last_two_bytes_context,
                            short_term_memory_.predictions, 0.005, false);
  models_.push_back(std::unique_ptr<Model>(mixer3));
  Mixer* second_layer = new Mixer(
      short_term_memory_, long_term_memory_, short_term_memory_.always_zero,
      short_term_memory_.mixer_outputs, 0.005, true);
  models_.push_back(std::unique_ptr<Model>(second_layer));
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
