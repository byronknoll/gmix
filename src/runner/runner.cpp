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
  printf("Without pretrained model:\n");
  printf("    Compress: gmix -c [input] [output]\n");
  printf("    Decompress: gmix -d [input] [output]\n");
  printf("    Train: gmix -t [training file] [test file]\n");
  printf("With pretrained model:\n");
  printf("    Compress: gmix -c [checkpoint_path] [input] [output]\n");
  printf("    Decompress: gmix -d [checkpoint_path] [input] [output]\n");
  printf("    Train: gmix -t [checkpoint_path] [training file] [test file]\n");
  printf(
      "    Generate: gmix -g [checkpoint_path] [prompt file] [output] "
      "[output_size] "
      "[temperature]\n");
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
    if (argc != 7) {
      printf("Wrong number of arguments.\n");
      return Help();
    }
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
  if (argc != 4 && argc != 5) {
    printf("Wrong number of arguments.\n");
    return Help();
  }

  std::string checkpoing_path = "";
  std::string input_path = argv[2];
  std::string output_path = argv[3];

  if (argc == 5) {
    checkpoing_path = argv[2];
    input_path = argv[3];
    output_path = argv[4];
  }

  unsigned long long input_bytes = 0, output_bytes = 0;

  if (argv[1][1] == 't') {
    if (!runner_utils::RunTraining(checkpoing_path, input_path, output_path,
                                   &input_bytes, &output_bytes)) {
      return Help();
    }
  } else if (argv[1][1] == 'c') {
    if (!runner_utils::RunCompression(checkpoing_path, input_path, output_path,
                                      &input_bytes, &output_bytes)) {
      return Help();
    }
  } else if (argv[1][1] == 'd') {
    if (!runner_utils::RunDecompression(checkpoing_path, input_path,
                                        output_path, &input_bytes,
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
