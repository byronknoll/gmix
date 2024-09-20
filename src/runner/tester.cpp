#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>

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

void CompressWithRestart(unsigned long long input_bytes, std::ifstream* is,
                         std::ofstream* os, unsigned long long* output_bytes) {
  Predictor* p = new Predictor();
  Encoder e(os, p);
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
      // Checkpoint and restart.
      p->WriteCheckpoint("data/checkpoint");
      delete p;
      p = new Predictor();
      p->ReadCheckpoint("data/checkpoint");
      e.SetPredictor(p);
    }
  }
  e.Flush();
  delete p;
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
  CompressWithRestart(*input_bytes, &data_in, &data_out, output_bytes);
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

void TestDecompression() {
  printf("TestDecompression:\n");
  unsigned long long in, out;
  runner_utils::RunDecompression("data/mid", "data/end", &in, &out);
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
  TestDecompression();
  if (!CompareFiles("./tester", "data/end")) return Fail();
  TestGeneration();
  printf("Tests passed.\n");
  return 0;
}
