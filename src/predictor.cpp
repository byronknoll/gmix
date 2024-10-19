#include "predictor.h"

#include <iomanip>
#include <numeric>

#include "contexts/basic-contexts.h"
#include "mixer/mixer.h"
#include "models/indirect.h"
#include "models/lstm-model.h"
#include "models/match.h"
#include "models/mod_ppmd.h"

Predictor::Predictor() : sigmoid_(100001), short_term_memory_(sigmoid_) {
  srand(0xDEADBEEF);
  AddModel(new BasicContexts());
  AddIndirect();
  AddModel(new PPMD::ModPPMD(short_term_memory_, long_term_memory_, 20, 1000));
  AddModel(new LstmModel(short_term_memory_, long_term_memory_));
  AddMatch();
  AddMixers();
  short_term_memory_.predictions.resize(short_term_memory_.num_predictions);
  short_term_memory_.predictions = 0.5;
  short_term_memory_.mixer_outputs.resize(short_term_memory_.num_mixers - 1);
  short_term_memory_.mixer_outputs = 0.5;
  short_term_memory_.entropy.resize(short_term_memory_.num_predictions +
                                    short_term_memory_.num_mixers);
  short_term_memory_.entropy = -1;
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

void Predictor::AddIndirect() {
  float learning_rate = 1.0f / 200;
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.last_byte_context,
                        "Indirect(1 byte)"));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.last_two_bytes_context,
                        "Indirect(2 bytes)"));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.last_three_bytes_15_bit_hash,
                        "Indirect(3 byte hash)"));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.last_four_bytes_15_bit_hash,
                        "Indirect(4 byte hash)"));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.last_five_bytes_15_bit_hash,
                        "Indirect(5 byte hash)"));
}

void Predictor::AddMatch() {
  int limit = 200;
  AddModel(new Match(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_two_bytes_context, limit,
                     "Match(2 bytes)"));
  AddModel(new Match(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_three_bytes_context, limit,
                     "Match(3 bytes)"));
  AddModel(new Match(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_five_bytes_21_bit_hash, limit,
                     "Match(5 byte hash)"));
}

void Predictor::AddMixers() {
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_byte_context,
                     short_term_memory_.predictions, 0.005, false,
                     "Mixer(1 byte)"));
  AddModel(new Mixer(
      short_term_memory_, long_term_memory_, short_term_memory_.longest_match,
      short_term_memory_.predictions, 0.0005, false, "Mixer(longest match)"));
  AddModel(new Mixer(
      short_term_memory_, long_term_memory_, short_term_memory_.always_zero,
      short_term_memory_.predictions, 0.0005, false, "Mixer(no context)"));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_two_bytes_context,
                     short_term_memory_.predictions, 0.005, false,
                     "Mixer(2 bytes)"));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_three_bytes_15_bit_hash,
                     short_term_memory_.predictions, 0.005, false,
                     "Mixer(3 byte hash)"));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_four_bytes_15_bit_hash,
                     short_term_memory_.predictions, 0.005, false,
                     "Mixer(4 byte hash)"));

  AddModel(new Mixer(
      short_term_memory_, long_term_memory_, short_term_memory_.always_zero,
      short_term_memory_.mixer_outputs, 0.005, true, "Mixer(final layer)"));
}

float Predictor::Predict() {
  short_term_memory_.active_models.clear();
  if (sample_frequency_ > 0) {
    // To compute cross entropy, set "inactive" model predictions to 0.5.
    short_term_memory_.predictions = 0.5;
  }
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

void Predictor::Perceive(int bit) {
  short_term_memory_.new_bit = bit;
  if (sample_frequency_ > 0) RunAnalysis(bit);
}

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

void Predictor::EnableAnalysis(int sample_frequency) {
  sample_frequency_ = sample_frequency;
  std::ofstream entropy("entropy.tsv", std::ios::out);
  std::ofstream memory("memory.tsv", std::ios::out);
  entropy << "bits seen";
  memory << "bits seen";
  for (int i = 0; i < short_term_memory_.model_descriptions.size(); ++i) {
    entropy << "\t" << short_term_memory_.model_descriptions[i];
    memory << "\t" << short_term_memory_.model_descriptions[i];
  }
  entropy << std::endl;
  memory << std::endl;
}

void Predictor::RunAnalysis(int bit) {
  for (int i = 0; i < short_term_memory_.entropy.size(); ++i) {
    float entropy = 0;
    float prob = 0.5;
    if (i < short_term_memory_.num_predictions) {
      prob = Sigmoid::Logistic(short_term_memory_.predictions[i]);
    } else if (i != short_term_memory_.entropy.size() - 1) {
      prob = Sigmoid::Logistic(
          short_term_memory_
              .mixer_outputs[i - short_term_memory_.num_predictions]);
    } else {
      prob = Sigmoid::Logistic(short_term_memory_.final_mixer_output);
    }
    float eps = 0.01;
    if (prob < eps)
      prob = eps;
    else if (prob > 1 - eps)
      prob = 1 - eps;
    if (bit)
      entropy = log2(prob);
    else
      entropy = log2(1 - prob);
    double alpha = 0.00001;
    short_term_memory_.entropy[i] =
        (1 - alpha) * short_term_memory_.entropy[i] + alpha * entropy;
  }
  if (short_term_memory_.bits_seen % sample_frequency_ == 0 &&
      short_term_memory_.bits_seen > 0) {
    std::ofstream entropy_file("entropy.tsv", std::ios::app);
    std::ofstream memory_file("memory.tsv", std::ios::app);
    entropy_file << short_term_memory_.bits_seen;
    memory_file << short_term_memory_.bits_seen;
    for (int i = 0; i < short_term_memory_.entropy.size(); ++i) {
      entropy_file << std::fixed << std::setprecision(5) << "\t"
                   << -short_term_memory_.entropy[i];
      if (i < short_term_memory_.num_predictions) {
        memory_file << "\t"
                    << short_term_memory_.prediction_index_to_model_ptr[i]
                           ->GetMemoryUsage(short_term_memory_,
                                            long_term_memory_);
      } else {
        memory_file
            << "\t"
            << short_term_memory_
                   .mixer_index_to_model_ptr[i -
                                             short_term_memory_.num_predictions]
                   ->GetMemoryUsage(short_term_memory_, long_term_memory_);
      }
    }
    entropy_file << std::endl;
    memory_file << std::endl;
  }
}