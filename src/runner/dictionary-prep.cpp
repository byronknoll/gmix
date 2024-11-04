#include <stdio.h>
#include <string.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

#include "../preprocess/dictionary.h"

int Help() {
  printf("This tool runs dictionary preprocessing.\n");
  printf("Encode: dictionary-prep -e [dictionary] [input] [output]\n");
  printf("Decode: dictionary-prep -d [dictionary] [input] [output]\n");
  return -1;
}

int main(int argc, char* argv[]) {
  if (argc != 5 || strlen(argv[1]) != 2 || argv[1][0] != '-' ||
      (argv[1][1] != 'e' && argv[1][1] != 'd')) {
    return Help();
  }

  clock_t start = clock();

  std::string dictionary_path = argv[2];
  std::string input_path = argv[3];
  std::string output_path = argv[4];

  FILE* dictionary = fopen(argv[2], "rb");
  if (!dictionary) return Help();

  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return Help();
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return Help();

  unsigned long long input_bytes = 0, output_bytes = 0;
  fseek(data_in, 0L, SEEK_END);
  input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);

  if (argv[1][1] == 'e') {
    preprocessor::Dictionary dict(dictionary, true, false);
    dict.Encode(data_in, input_bytes, data_out);
  } else {
    preprocessor::Dictionary dict(dictionary, false, true);
    dict.Decode(data_in, data_out);
  }
  output_bytes = ftell(data_out);
  fclose(dictionary);
  fclose(data_in);
  fclose(data_out);

  printf("\r%lld bytes -> %lld bytes in %1.2f s.\n", input_bytes, output_bytes,
         ((double)clock() - start) / CLOCKS_PER_SEC);

  return 0;
}
