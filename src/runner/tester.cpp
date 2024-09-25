#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>

#include "../coder/decoder.h"
#include "../coder/encoder.h"
#include "runner-utils.h"

bool CompareFiles(const std::string& p1, const std::string& p2) {
  std::ifstream f1(p1, std::ifstream::binary | std::ifstream::ate);
  std::ifstream f2(p2, std::ifstream::binary | std::ifstream::ate);

  if (f1.fail() || f2.fail()) {
    return false;  // file problem
  }

  if (f1.tellg() != f2.tellg()) {
    return false;  // size mismatch
  }

  // Seek back to beginning and use std::equal to compare contents.
  f1.seekg(0, std::ifstream::beg);
  f2.seekg(0, std::ifstream::beg);
  return std::equal(std::istreambuf_iterator<char>(f1.rdbuf()),
                    std::istreambuf_iterator<char>(),
                    std::istreambuf_iterator<char>(f2.rdbuf()));
}

void CompressFirstHalf(unsigned long long input_bytes, std::ifstream* is,
                       std::ofstream* os, unsigned long long* output_bytes) {
  Predictor p;
  Encoder e(os, &p);
  unsigned long long percent = 1 + (input_bytes / 10000);
  runner_utils::ClearOutput();
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
    if (pos == input_bytes / 2) {
      // Checkpoint and return.
      p.WriteCheckpoint("data/checkpoint");
      e.WriteCheckpoint("data/checkpoint2");
      return;
    }
  }
}

void CompressSecondHalf(unsigned long long input_bytes, std::ifstream* is,
                        std::ofstream* os, unsigned long long* output_bytes) {
  Predictor p;
  p.ReadCheckpoint("data/checkpoint");
  Encoder e(os, &p);
  e.ReadCheckpoint("data/checkpoint2");
  p.WriteCheckpoint("data/checkpoint3");
  unsigned long long percent = 1 + (input_bytes / 10000);
  runner_utils::ClearOutput();
  for (unsigned long long pos = (input_bytes / 2) + 1; pos < input_bytes;
       ++pos) {
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

bool RunCompressionWithRestart(const std::string& input_path,
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

  runner_utils::WriteHeader(*input_bytes, &data_out);
  CompressFirstHalf(*input_bytes, &data_in, &data_out, output_bytes);
  CompressSecondHalf(*input_bytes, &data_in, &data_out, output_bytes);
  data_in.close();
  data_out.close();
  return true;
}

void DecompressFirstHalf(unsigned long long output_length, std::ifstream* is,
                         std::ofstream* os) {
  Predictor p;
  Decoder d(is, &p);
  unsigned long long percent = 1 + (output_length / 10000);
  runner_utils::ClearOutput();
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

    if (pos == output_length / 2) {
      // Checkpoint and return.
      p.WriteCheckpoint("data/checkpoint");
      d.WriteCheckpoint("data/checkpoint2");
      return;
    }
  }
}

void DecompressSecondHalf(unsigned long long output_length, std::ifstream* is,
                          std::ofstream* os) {
  Predictor p;
  p.ReadCheckpoint("data/checkpoint");
  Decoder d(is, &p, true);
  d.ReadCheckpoint("data/checkpoint2");
  unsigned long long percent = 1 + (output_length / 10000);
  runner_utils::ClearOutput();
  for (unsigned long long pos = (output_length / 2) + 1; pos < output_length;
       ++pos) {
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

bool RunDecompressionWithRestart(const std::string& input_path,
                                 const std::string& output_path,
                                 unsigned long long* input_bytes,
                                 unsigned long long* output_bytes) {
  std::ifstream data_in(input_path, std::ios::in | std::ios::binary);
  if (!data_in.is_open()) return false;

  data_in.seekg(0, std::ios::end);
  *input_bytes = data_in.tellg();
  data_in.seekg(0, std::ios::beg);
  runner_utils::ReadHeader(&data_in, output_bytes);

  std::ofstream data_out(output_path, std::ios::out | std::ios::binary);
  if (!data_out.is_open()) return false;

  DecompressFirstHalf(*output_bytes, &data_in, &data_out);
  DecompressSecondHalf(*output_bytes, &data_in, &data_out);
  data_in.close();
  data_out.close();
  return true;
}

void TestCompression() {
  printf("TestCompression:\n");
  unsigned long long in, out;
  runner_utils::RunCompression("./tester", "data/mid", &in, &out);
  printf("\n");
}

void TestCompressionWithRestart() {
  printf("TestCompressionWithRestart:\n");
  unsigned long long in, out;
  RunCompressionWithRestart("./tester", "data/mid2", &in, &out);
  printf("\n");
}

void TestDecompressionWithRestart() {
  printf("TestDecompressionWithRestart:\n");
  unsigned long long in, out;
  RunDecompressionWithRestart("data/mid", "data/end", &in, &out);
  printf("\n");
}

void TestGeneration() {
  printf("TestGeneration:\n");
  runner_utils::RunGeneration("data/mid", "data/end", 100);
  printf("\n");
}

int Fail() {
  printf("Test failed.\n");
  return -1;
}

int main(int argc, char* argv[]) {
  TestCompression();
  TestCompressionWithRestart();
  if (!CompareFiles("data/mid", "data/mid2")) return Fail();
  if (!CompareFiles("data/checkpoint.long", "data/checkpoint3.long")) return Fail();
  if (!CompareFiles("data/checkpoint.short", "data/checkpoint3.short")) return Fail();
  TestDecompressionWithRestart();
  if (!CompareFiles("./tester", "data/end")) return Fail();
  TestGeneration();
  printf("Tests passed.\n");
  return 0;
}
