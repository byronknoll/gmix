#include "predictor.h"

#include <filesystem>
#include <iomanip>
#include <numeric>

#include "contexts/basic-contexts.h"
#include "contexts/indirect-hash.h"
#include "contexts/interval-context.h"
#include "mixer/mixer.h"
#include "models/indirect.h"
#include "models/lstm-model.h"
#include "models/match.h"
#include "models/mod_ppmd.h"

Predictor::Predictor() {
  srand(0xDEADBEEF);
  AddModel(new BasicContexts());
  AddIntervalContexts();
  AddIndirect();
  AddModel(
      new PPMD::ModPPMD(short_term_memory_, long_term_memory_, 20, 2000, true));
  AddModel(new LstmModel(short_term_memory_, long_term_memory_, true));
  AddMatch();
  AddDoubleIndirect();
  AddMixers();
  short_term_memory_.predictions.resize(short_term_memory_.num_predictions);
  short_term_memory_.predictions = 0;
  short_term_memory_.mixer_layer0_outputs.resize(
      short_term_memory_.num_layer0_mixers);
  short_term_memory_.mixer_layer0_outputs = 0;
  short_term_memory_.mixer_layer1_outputs.resize(
      short_term_memory_.num_layer1_mixers);
  short_term_memory_.mixer_layer1_outputs = 0;
  short_term_memory_.entropy.resize(
      short_term_memory_.model_descriptions.size());
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

void Predictor::AddIntervalContexts() {
  std::vector<int> map(256, 0);
  for (int i = 0; i < 256; ++i) {
    map[i] = i / 16;  // states = 0-15 (four bits)
  }
  AddModel(new IntervalContext(map, 4, short_term_memory_.interval_16_4));
  AddModel(new IntervalContext(map, 8, short_term_memory_.interval_16_8));
  AddModel(new IntervalContext(map, 12, short_term_memory_.interval_16_12));

  for (int i = 0; i < 256; ++i) {
    map[i] = i / 32;  // states = 0-7 (three bits)
  }
  AddModel(new IntervalContext(map, 3, short_term_memory_.interval_32_3));
  AddModel(new IntervalContext(map, 6, short_term_memory_.interval_32_6));
  AddModel(new IntervalContext(map, 9, short_term_memory_.interval_32_9));
  AddModel(new IntervalContext(map, 12, short_term_memory_.interval_32_12));

  for (int i = 0; i < 256; ++i) {
    map[i] = i / 64;  // states = 0-3 (two bits)
  }
  AddModel(new IntervalContext(map, 4, short_term_memory_.interval_64_4));
  AddModel(new IntervalContext(map, 8, short_term_memory_.interval_64_8));
  AddModel(new IntervalContext(map, 12, short_term_memory_.interval_64_12));
}

void Predictor::AddIndirect() {
  float learning_rate = 0.02;
  bool enable_analysis = true;
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.last_byte, "Indirect(1 byte)",
                        enable_analysis));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.last_two_bytes_context,
                        "Indirect(2 bytes)", enable_analysis));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.last_three_bytes_15_bit_hash,
                        "Indirect(3_15 byte hash)", enable_analysis));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.last_three_bytes_16_bit_hash,
                        "Indirect(3_16 byte hash)", enable_analysis));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.last_four_bytes_15_bit_hash,
                        "Indirect(4 byte hash)", enable_analysis));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.last_five_bytes_15_bit_hash,
                        "Indirect(5 byte hash)", enable_analysis));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.last_six_bytes_15_bit_hash,
                        "Indirect(6 byte hash)", enable_analysis));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.second_last_byte,
                        "Indirect(2nd last byte)", enable_analysis));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.third_last_byte,
                        "Indirect(3rd last byte)", enable_analysis));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.fourth_last_byte,
                        "Indirect(4th last byte)", enable_analysis));
}

void Predictor::AddMatch() {
  int limit = 400;
  bool enable_analysis = true;
  AddModel(new Match(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_byte, limit, "Match(1 byte)",
                     enable_analysis));
  AddModel(new Match(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_two_bytes_context, limit,
                     "Match(2 bytes)", enable_analysis));
  AddModel(new Match(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_three_bytes_context, limit,
                     "Match(3 bytes)", enable_analysis));
  AddModel(new Match(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_four_bytes_21_bit_hash, limit,
                     "Match(4 byte hash)", enable_analysis));
  AddModel(new Match(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_five_bytes_21_bit_hash, limit,
                     "Match(5 byte hash)", enable_analysis));
  AddModel(new Match(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_six_bytes_21_bit_hash, limit,
                     "Match(6 byte hash)", enable_analysis));
}

void Predictor::AddDoubleIndirect() {
  float learning_rate = 1.0f / 200;
  bool enable_analysis = true;
  AddModel(new IndirectHash(1, 8, 1, 8, short_term_memory_.indirect_1_8_1_8));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.indirect_1_8_1_8,
                        "Indirect(indirect_1_8_1_8)", enable_analysis));
  AddModel(new IndirectHash(1, 8, 2, 16, short_term_memory_.indirect_1_8_2_16));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.indirect_1_8_2_16,
                        "Indirect(indirect_1_8_2_16)", enable_analysis));
  AddModel(new IndirectHash(1, 8, 3, 15, short_term_memory_.indirect_1_8_3_15));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.indirect_1_8_3_15,
                        "Indirect(indirect_1_8_3_15)", enable_analysis));
  AddModel(new IndirectHash(2, 16, 1, 8, short_term_memory_.indirect_2_16_1_8));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.indirect_2_16_1_8,
                        "Indirect(indirect_2_16_1_8)", enable_analysis));
  AddModel(
      new IndirectHash(2, 16, 2, 16, short_term_memory_.indirect_2_16_2_16));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.indirect_2_16_2_16,
                        "Indirect(indirect_2_16_2_16)", enable_analysis));
  AddModel(new IndirectHash(3, 24, 1, 8, short_term_memory_.indirect_3_24_1_8));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.indirect_3_24_1_8,
                        "Indirect(indirect_3_24_1_8)", enable_analysis));
  AddModel(
      new IndirectHash(4, 24, 2, 15, short_term_memory_.indirect_4_24_2_15));
  AddModel(new Indirect(short_term_memory_, long_term_memory_, learning_rate,
                        short_term_memory_.indirect_4_24_2_15,
                        "Indirect(indirect_4_24_2_15)", enable_analysis));
}

void Predictor::AddMixers() {
  bool enable_analysis = false;
  // First layer.
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_byte, 0.005, 0,
                     "Mixer0(last byte)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.fourth_last_byte, 0.005, 0,
                     "Mixer0(4th last byte)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.second_last_plus_recent, 0.003, 0,
                     "Mixer0(2nd last + recent)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_four_bytes_15_bit_hash, 0.005, 0,
                     "Mixer0(4 byte hash)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.indirect_3_24_1_8, 0.005, 0,
                     "Mixer0(indirect_3_24_1_8)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.second_last_byte, 0.005, 0,
                     "Mixer0(2nd last byte)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.longest_match, 0.0005, 0,
                     "Mixer0(longest match)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_two_bytes_context, 0.005, 0,
                     "Mixer0(2 bytes)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.third_last_byte, 0.005, 0,
                     "Mixer0(3rd last byte)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_three_bytes_15_bit_hash, 0.005, 0,
                     "Mixer0(3 byte hash)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_byte, 0.001, 0,
                     "Mixer0(last byte)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_byte_plus_recent, 0.005, 0,
                     "Mixer0(last byte + recent)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.interval_16_4, 0.005, 0,
                     "Mixer0(interval_16_4)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.interval_16_8, 0.005, 0,
                     "Mixer0(interval_16_8)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.interval_16_12, 0.005, 0,
                     "Mixer0(interval_16_12)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.interval_32_3, 0.005, 0,
                     "Mixer0(interval_32_3)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.interval_32_6, 0.005, 0,
                     "Mixer0(interval_32_6)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.interval_32_9, 0.005, 0,
                     "Mixer0(interval_32_9)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.interval_32_12, 0.005, 0,
                     "Mixer0(interval_32_12)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.interval_64_4, 0.005, 0,
                     "Mixer0(interval_64_4)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.interval_64_8, 0.005, 0,
                     "Mixer0(interval_64_8)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.interval_64_12, 0.005, 0,
                     "Mixer0(interval_64_12)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.always_zero, 0.0005, 0,
                     "Mixer0(no context)", enable_analysis));

  // Second layer.
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.second_last_byte, 0.005, 1,
                     "Mixer1(2nd last byte)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.always_zero, 0.005, 1,
                     "Mixer1(no context)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.bit_context, 0.005, 1,
                     "Mixer1(recent_bits)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.third_last_byte, 0.005, 1,
                     "Mixer1(3rd last byte)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.last_byte, 0.005, 1,
                     "Mixer1(last byte)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.bit_context, 0.00001, 1,
                     "Mixer1(recent_bits)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.longest_match, 0.0005, 1,
                     "Mixer1(longest match)", enable_analysis));
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.always_zero, 0.0005, 1,
                     "Mixer1(no context)", enable_analysis));

  // Final layer.
  enable_analysis = true;
  AddModel(new Mixer(short_term_memory_, long_term_memory_,
                     short_term_memory_.always_zero, 0.0005, 2,
                     "Mixer(final layer)", enable_analysis));
}

float Predictor::Predict() {
  short_term_memory_.active_models.clear();
  if (sample_frequency_ > 0) {
    // To compute cross entropy, set "inactive" model predictions to 0.5.
    short_term_memory_.predictions = 0;
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
  std::filesystem::create_directory("analysis");
  std::ofstream entropy("analysis/entropy.tsv", std::ios::out);
  std::ofstream memory("analysis/memory.tsv", std::ios::out);
  entropy << "bits seen";
  memory << "bits seen";
  for (int i = 0; i < short_term_memory_.model_descriptions.size(); ++i) {
    if (!short_term_memory_.model_enable_analysis[i]) continue;
    entropy << "\t" << short_term_memory_.model_descriptions[i];
    memory << "\t" << short_term_memory_.model_descriptions[i];
  }
  memory << "\tmatch history";
  entropy << std::endl;
  memory << std::endl;
}

void Predictor::UpdateEntropy(int bit, int index) {
  float entropy = 0;
  float prob = 0.5;
  if (index < short_term_memory_.num_predictions) {
    prob = Sigmoid::Logistic(short_term_memory_.predictions[index]);
  } else if (index - short_term_memory_.num_predictions <
             short_term_memory_.num_layer0_mixers) {
    prob = Sigmoid::Logistic(
        short_term_memory_
            .mixer_layer0_outputs[index - short_term_memory_.num_predictions]);
  } else if (index != short_term_memory_.entropy.size() - 1) {
    int mixer_index = index - short_term_memory_.num_predictions -
                      short_term_memory_.num_layer0_mixers;
    prob =
        Sigmoid::Logistic(short_term_memory_.mixer_layer1_outputs[mixer_index]);
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
  short_term_memory_.entropy[index] =
      (1 - alpha) * short_term_memory_.entropy[index] + alpha * entropy;
}

void Predictor::RunAnalysis(int bit) {
  for (int i = 0; i < short_term_memory_.model_enable_analysis.size(); ++i) {
    if (!short_term_memory_.model_enable_analysis[i]) continue;
    UpdateEntropy(bit, i);
  }
  if (short_term_memory_.bits_seen % sample_frequency_ == 0 &&
      short_term_memory_.bits_seen > 0) {
    std::ofstream entropy_file("analysis/entropy.tsv", std::ios::app);
    std::ofstream memory_file("analysis/memory.tsv", std::ios::app);
    entropy_file << short_term_memory_.bits_seen;
    memory_file << short_term_memory_.bits_seen;
    for (int i = 0; i < short_term_memory_.model_enable_analysis.size(); ++i) {
      if (!short_term_memory_.model_enable_analysis[i]) continue;
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
    memory_file << "\t" << long_term_memory_.history.size();
    entropy_file << std::endl;
    memory_file << std::endl;
  }
}