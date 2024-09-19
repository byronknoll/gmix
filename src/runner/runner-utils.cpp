#include "runner-utils.h"

#include <stdio.h>
#include <string.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

#include "../coder/decoder.h"
#include "../coder/encoder.h"
#include "../predictor.h"

namespace runner_utils {
inline float Rand() {
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
  Encoder e(os, p);
  unsigned long long percent = 1 + (input_bytes / 10000);
  ClearOutput();
  for (unsigned long long pos = 0; pos < input_bytes; ++pos) {
    char c = is->get();
    for (int j = 7; j >= 0; --j) {
      e.Encode((c >> j) & 1);
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

bool RunCompression(const std::string& input_path,
                    const std::string& output_path,
                    unsigned long long* input_bytes,
                    unsigned long long* output_bytes) {
  std::ifstream data_in(input_path, std::ios::in | std::ios::binary);
  if (!data_in.is_open()) return false;

  data_in.seekg(0, std::ios::end);
  *input_bytes = data_in.tellg();
  data_in.seekg(0, std::ios::beg);

  std::ofstream data_out(output_path, std::ios::out | std::ios::binary);
  if (!data_out.is_open()) return false;

  WriteHeader(*input_bytes, &data_out);
  Predictor p;
  Compress(*input_bytes, &data_in, &data_out, output_bytes, &p);
  data_in.close();
  data_out.close();
  return true;
}

bool RunDecompression(const std::string& input_path,
                      const std::string& output_path,
                      unsigned long long* input_bytes,
                      unsigned long long* output_bytes) {
  std::ifstream data_in(input_path, std::ios::in | std::ios::binary);
  if (!data_in.is_open()) return false;

  data_in.seekg(0, std::ios::end);
  *input_bytes = data_in.tellg();
  data_in.seekg(0, std::ios::beg);
  ReadHeader(&data_in, output_bytes);
  Predictor p;

  std::ofstream data_out(output_path, std::ios::out | std::ios::binary);
  if (!data_out.is_open()) return false;

  Decompress(*output_bytes, &data_in, &data_out, &p);
  data_in.close();
  data_out.close();
  return true;
}

bool RunGeneration(const std::string& input_path,
                   const std::string& output_path, int output_size) {
  std::ifstream data_in(input_path, std::ios::in | std::ios::binary);
  if (!data_in.is_open()) return false;

  std::ofstream data_out(output_path, std::ios::out | std::ios::binary);
  if (!data_out.is_open()) return false;

  data_in.seekg(0, std::ios::end);
  unsigned long long input_bytes = data_in.tellg();
  data_in.seekg(0, std::ios::beg);

  Predictor p;
  float prob = p.Predict();
  double entropy = 0;
  unsigned long long percent = 1 + (input_bytes / 100);
  for (unsigned int pos = 0; pos < input_bytes; ++pos) {
    int c = data_in.get();
    for (int j = 7; j >= 0; --j) {
      int bit = (c >> j) & 1;
      if (bit)
        entropy += log2(prob);
      else
        entropy += log2(1 - prob);
      p.Perceive(bit);
      p.Learn();
      prob = p.Predict();
    }
    if (pos % percent == 0) {
      printf("\rtraining: %lld%%", pos / percent);
      fflush(stdout);
    }
  }
  entropy = -entropy / input_bytes;
  printf("\rcross entropy: %.4f\n", entropy);

  data_in.close();

  percent = 1 + (output_size / 100);
  for (int i = 0; i < output_size; ++i) {
    int byte = 1;
    while (byte < 256) {
      int bit = 0;
      float r = Rand();
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

}  // namespace runner_utils