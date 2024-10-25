#include <stdio.h>
#include <string.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

#include "../coder/decoder.h"
#include "../coder/encoder.h"
#include "../predictor.h"
#include "runner-utils.h"

int Help() {
  printf("gmix version 1\n");
  printf("Compress: gmix -c [input] [output]\n");
  printf("Decompress: gmix -d [input] [output]\n");
  printf(
      "Generate: gmix -g [checkpoint_path] [prompt_path] [output] "
      "[output_size] "
      "[temperature]\n");
  printf("Train: gmix -t [training file] [test file]\n");
  return -1;
}

int main(int argc, char* argv[]) {
  if (argc < 4 || argc > 7 || strlen(argv[1]) != 2 || argv[1][0] != '-' ||
      (argv[1][1] != 'c' && argv[1][1] != 'd' && argv[1][1] != 'g' &&
       argv[1][1] != 't')) {
    return Help();
  }
  srand(0xDEADBEEF);
  clock_t start = clock();

  if (argv[1][1] == 'g') {
    if (argc != 7) return Help();
    std::string checkpoint_path = argv[2];
    std::string prompt_path = argv[3];
    std::string output_path = argv[4];
    int output_size = std::stoi(argv[5]);
    float temperature = std::stof(argv[6]);

    if (!runner_utils::RunGeneration(checkpoint_path, prompt_path, output_path,
                                     output_size, temperature)) {
      return Help();
    }
    printf("%1.2f s.\n", ((double)clock() - start) / CLOCKS_PER_SEC);
    return 0;
  }
  if (argc != 4) return Help();

  std::string input_path = argv[2];
  std::string output_path = argv[3];

  unsigned long long input_bytes = 0, output_bytes = 0;

  if (argv[1][1] == 't') {
    if (!runner_utils::RunTraining(input_path, output_path, &input_bytes,
                                   &output_bytes)) {
      return Help();
    }
  } else if (argv[1][1] == 'c') {
    if (!runner_utils::RunCompression(input_path, output_path, &input_bytes,
                                      &output_bytes)) {
      return Help();
    }
  } else if (argv[1][1] == 'd') {
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
