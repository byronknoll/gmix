#include <stdio.h>
#include <string.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

#include "coder/decoder.h"
#include "coder/encoder.h"
#include "predictor.h"

int Help() {
  printf("gmix version 1\n");
  printf("Compress: gmix -c [input] [output]\n");
  printf("Decompress: gmix -d [input] [output]\n");
  return -1;
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

int main(int argc, char* argv[]) {
  if (argc != 4 || strlen(argv[1]) != 2 || argv[1][0] != '-' ||
      (argv[1][1] != 'c' && argv[1][1] != 'd')) {
    return Help();
  }

  clock_t start = clock();

  std::string input_path = argv[2];
  std::string output_path = argv[3];

  unsigned long long input_bytes = 0, output_bytes = 0;

  if (argv[1][1] == 'c') {
    if (!RunCompression(input_path, output_path, &input_bytes, &output_bytes)) {
      return Help();
    }
  } else {
    if (!RunDecompression(input_path, output_path, &input_bytes,
                          &output_bytes)) {
      return Help();
    }
  }

  printf("\r%lld bytes -> %lld bytes in %1.2f s.\n", input_bytes, output_bytes,
         ((double)clock() - start) / CLOCKS_PER_SEC);

  if (argv[1][1] == 'c') {
    double cross_entropy = output_bytes;
    cross_entropy /= input_bytes;
    cross_entropy *= 8;
    printf("cross entropy: %.3f\n", cross_entropy);
  }

  return 0;
}
