#include "runner-utils.h"

#include <stdio.h>
#include <string.h>

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <vector>

#include "../coder/decoder.h"
#include "../coder/encoder.h"
#include "../predictor.h"

namespace runner_utils {
float Rand() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void WriteHeader(unsigned long long length, std::ofstream* os) {
  for (int i = 4; i >= 0; --i) {
    char c = length >> (8 * i);
    os->put(c);
  }
}

void ReadHeader(std::ifstream* is, unsigned long long* length) {
  *length = 0;
  for (int i = 0; i <= 4; ++i) {
    *length <<= 8;
    unsigned char c = is->get();
    *length += c;
  }
}

void ClearOutput() {
  fprintf(stderr, "\r                     \r");
  fflush(stderr);
}

void Compress(unsigned long long input_bytes, std::ifstream* is,
              std::ofstream* os, unsigned long long* output_bytes,
              Predictor* p) {
  Encoder e(os);
  p->EnableAnalysis(8 * input_bytes / 1000);
  unsigned long long percent = 1 + (input_bytes / 10000);
  ClearOutput();
  for (unsigned long long pos = 0; pos < input_bytes; ++pos) {
    char c = is->get();
    for (int j = 7; j >= 0; --j) {
      int bit = (c >> j) & 1;
      float prediction = p->Predict();
      e.Encode(bit, prediction);
      p->Perceive(bit);
      p->Learn();
    }
    if (pos % percent == 0) {
      double frac = 100.0 * pos / input_bytes;
      fprintf(stderr, "\rprogress: %.2f%%", frac);
      fflush(stderr);
    }
  }
  e.Flush();
  *output_bytes = os->tellp();
}

void Decompress(unsigned long long output_length, std::ifstream* is,
                std::ofstream* os, Predictor* p) {
  Decoder d(is, p);
  unsigned long long percent = 1 + (output_length / 10000);
  ClearOutput();
  for (unsigned long long pos = 0; pos < output_length; ++pos) {
    int byte = 1;
    while (byte < 256) {
      byte += byte + d.Decode();
    }
    os->put(byte);
    if (pos % percent == 0) {
      double frac = 100.0 * pos / output_length;
      fprintf(stderr, "\rprogress: %.2f%%", frac);
      fflush(stderr);
    }
  }
}

bool RunCompression(const std::string& checkpoint_path,
                    const std::string& input_path,
                    const std::string& output_path,
                    unsigned long long* input_bytes,
                    unsigned long long* output_bytes) {
  std::ifstream data_in(input_path, std::ios::in | std::ios::binary);
  if (!data_in.is_open()) {
    printf("Can not open: %s\n", input_path.c_str());
    return false;
  }

  data_in.seekg(0, std::ios::end);
  *input_bytes = data_in.tellg();
  data_in.seekg(0, std::ios::beg);

  std::ofstream data_out(output_path, std::ios::out | std::ios::binary);
  if (!data_out.is_open()) {
    printf("Can not open: %s\n", output_path.c_str());
    return false;
  }

  WriteHeader(*input_bytes, &data_out);
  Predictor p;
  if (!checkpoint_path.empty()) {
    printf("\rLoading checkpoint...");
    fflush(stdout);
    p.ReadCheckpoint(checkpoint_path);
    printf("\r                        ");
  }
  Compress(*input_bytes, &data_in, &data_out, output_bytes, &p);
  data_in.close();
  data_out.close();
  return true;
}

bool RunDecompression(const std::string& checkpoint_path,
                      const std::string& input_path,
                      const std::string& output_path,
                      unsigned long long* input_bytes,
                      unsigned long long* output_bytes) {
  std::ifstream data_in(input_path, std::ios::in | std::ios::binary);
  if (!data_in.is_open()) {
    printf("Can not open: %s\n", input_path.c_str());
    return false;
  }

  data_in.seekg(0, std::ios::end);
  *input_bytes = data_in.tellg();
  data_in.seekg(0, std::ios::beg);
  ReadHeader(&data_in, output_bytes);
  Predictor p;
  if (!checkpoint_path.empty()) {
    printf("\rLoading checkpoint...");
    fflush(stdout);
    p.ReadCheckpoint(checkpoint_path);
    printf("\r                        ");
  }

  std::ofstream data_out(output_path, std::ios::out | std::ios::binary);
  if (!data_out.is_open()) {
    printf("Can not open: %s\n", output_path.c_str());
    return false;
  }

  Decompress(*output_bytes, &data_in, &data_out, &p);
  data_in.close();
  data_out.close();
  return true;
}

bool RunGeneration(const std::string& checkpoint_path,
                   const std::string& prompt_path,
                   const std::string& output_path, int output_size,
                   float temperature) {
  std::ifstream data_in(prompt_path, std::ios::in | std::ios::binary);
  if (!data_in.is_open()) {
    printf("Can not open: %s\n", prompt_path.c_str());
    return false;
  }
  std::ofstream data_out(output_path, std::ios::out | std::ios::binary);
  if (!data_out.is_open()) {
    printf("Can not open: %s\n", output_path.c_str());
    return false;
  }
  if (temperature < 0.001) temperature = 0.001;

  Predictor p;
  printf("\rLoading checkpoint...");
  fflush(stdout);
  p.ReadCheckpoint(checkpoint_path);
  p.EnableAnalysis(8 * output_size / 1000);

  printf("\r                        ");
  printf("\rRunning prompt...");
  fflush(stdout);
  data_in.seekg(0, std::ios::end);
  unsigned long long input_bytes = data_in.tellg();
  data_in.seekg(0, std::ios::beg);
  // Skip the last byte ('\n' for text files).
  for (unsigned long long pos = 0; pos < input_bytes - 1; ++pos) {
    char c = data_in.get();
    for (int j = 7; j >= 0; --j) {
      p.Predict();
      p.Perceive((c >> j) & 1);
      p.Learn();
    }
  }
  printf("\r                        ");

  unsigned long long percent = 1 + (output_size / 100);
  float prob = p.Predict();
  for (int i = 0; i < output_size; ++i) {
    int byte = 1;
    while (byte < 256) {
      int bit = 0;
      float r = Rand();
      prob = Sigmoid::Logistic(Sigmoid::Logit(prob) / temperature);
      if (r < prob) bit = 1;
      byte += byte + bit;
      p.Perceive(bit);
      prob = p.Predict();
    }
    data_out.put(byte);
    if (i % percent == 0) {
      printf("\rgeneration: %lld%%", i / percent);
      fflush(stdout);
    }
  }
  printf("\rgeneration: 100%%\n");

  data_out.close();

  return true;
}

bool RunTraining(const std::string& checkpoint_path,
                 const std::string& train_path, const std::string& test_path,
                 unsigned long long* input_bytes,
                 unsigned long long* output_bytes) {
  std::ifstream data_train(train_path, std::ios::in | std::ios::binary);
  if (!data_train.is_open()) {
    printf("Can not open: %s\n", train_path.c_str());
    return false;
  }

  std::ifstream data_test(test_path, std::ios::in | std::ios::binary);
  if (!data_test.is_open()) {
    printf("Can not open: %s\n", test_path.c_str());
    return false;
  }

  data_train.seekg(0, std::ios::end);
  *input_bytes = data_train.tellg();
  data_train.seekg(0, std::ios::beg);

  data_test.seekg(0, std::ios::end);
  unsigned long long test_bytes = data_test.tellg();
  data_test.seekg(0, std::ios::beg);

  std::filesystem::create_directory("analysis");
  std::ofstream metrics("analysis/training.tsv", std::ios::out);
  metrics << "bytes\ttrain_entropy\ttest_entropy" << std::endl;

  std::filesystem::create_directory("data");
  std::ofstream data_out("data/tmp", std::ios::out | std::ios::binary);
  if (!data_out.is_open()) {
    printf("Can not open: data/tmp\n");
    return false;
  }

  WriteHeader(*input_bytes, &data_out);

  Predictor p;
  if (!checkpoint_path.empty()) {
    printf("\rLoading checkpoint...");
    fflush(stdout);
    p.ReadCheckpoint(checkpoint_path);
    printf("\r                        ");
  }
  Encoder e(&data_out);
  p.EnableAnalysis(8 * (*input_bytes) / 1000);
  float prob;
  double train_entropy = 0;
  unsigned long long percent = 1 + ((*input_bytes) / 100);
  for (unsigned int pos = 0; pos < *input_bytes; ++pos) {
    int c = data_train.get();
    for (int j = 7; j >= 0; --j) {
      prob = p.Predict();
      int bit = (c >> j) & 1;
      e.Encode(bit, prob);
      if (bit)
        train_entropy += log2(prob);
      else
        train_entropy += log2(1 - prob);
      p.Perceive(bit);
      p.Learn();
    }
    if (pos % percent == 0) {
      printf("\rtraining: %lld%%", pos / percent);
      fflush(stdout);
      if (pos == 0) continue;
      if ((pos / percent % 2) != 0) continue;

      Predictor p2;
      p2.Copy(p);
      data_test.seekg(0, std::ios::beg);
      double test_entropy = 0;
      float prob2 = prob;
      for (unsigned int pos2 = 0; pos2 < test_bytes; ++pos2) {
        int c2 = data_test.get();
        for (int j = 7; j >= 0; --j) {
          prob2 = p2.Predict();
          int bit = (c2 >> j) & 1;
          if (bit)
            test_entropy += log2(prob2);
          else
            test_entropy += log2(1 - prob2);
          p2.Perceive(bit);
          p2.Learn();
        }
      }
      metrics << std::fixed << std::setprecision(5) << pos << "\t"
              << -train_entropy / pos << "\t" << -test_entropy / test_bytes
              << std::endl;
    }
  }
  train_entropy = -train_entropy / *input_bytes;
  printf("\rtraining cross entropy: %.4f\n", train_entropy);
  e.Flush();
  *output_bytes = data_out.tellp();

  p.WriteCheckpoint("data/trained_checkpoint");

  return true;
}

}  // namespace runner_utils