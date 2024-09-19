#include "runner-utils.h"
#include "coder/encoder.h"

void TestCompression() {
  unsigned long long in, out;
  runner_utils::RunCompression("./tester", "data/mid", &in, &out);
  printf("\n");
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

void TestCompressionWithRestart() {
  unsigned long long in, out;
  RunCompressionWithRestart("./tester", "data/mid2", &in, &out);
  printf("\n");
}

void TestDecompression() {
  unsigned long long in, out;
  runner_utils::RunDecompression("data/mid", "data/end", &in, &out);
  runner_utils::RunDecompression("data/mid2", "data/end2", &in, &out);
  printf("\n");}

int main(int argc, char* argv[]) {
  TestCompression();
  TestCompressionWithRestart();
  TestDecompression();
  printf("Tests passed.\n");
  return 0;
}
