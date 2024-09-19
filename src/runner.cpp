#include <stdio.h>
#include <string.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

#include "coder/decoder.h"
#include "coder/encoder.h"
#include "predictor.h"
#include "runner-utils.h"

int Help() {
  printf("gmix version 1\n");
  printf("Compress: gmix -c [input] [output]\n");
  printf("Decompress: gmix -d [input] [output]\n");
  printf("Generate: gmix -g [input] [output] [output_size]\n");
  return -1;
}

int main(int argc, char* argv[]) {
  if (argc < 4 || argc > 5 || strlen(argv[1]) != 2 || argv[1][0] != '-' ||
      (argv[1][1] != 'c' && argv[1][1] != 'd' && argv[1][1] != 'g')) {
    return Help();
  }

  clock_t start = clock();

  std::string input_path = argv[2];
  std::string output_path = argv[3];

  if (argv[1][1] == 'g') {
    if (argc != 5) return Help();
    int output_size = std::stoi(argv[4]);
    if (!runner_utils::RunGeneration(input_path, output_path, output_size)) {
      return Help();
    }
    return 0;
  }
  if (argc != 4) return Help();

  unsigned long long input_bytes = 0, output_bytes = 0;

  if (argv[1][1] == 'c') {
    if (!runner_utils::RunCompression(input_path, output_path, &input_bytes,
                                      &output_bytes)) {
      return Help();
    }
  } else {
    if (!runner_utils::RunDecompression(input_path, output_path, &input_bytes,
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
