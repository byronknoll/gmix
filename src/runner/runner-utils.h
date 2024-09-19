#include <string.h>

#include <fstream>

#include "../predictor.h"

namespace runner_utils {
inline float Rand();

void WriteHeader(unsigned long long length, std::ofstream* os);

void ReadHeader(std::ifstream* is, unsigned long long* length);

void ClearOutput();

void Compress(unsigned long long input_bytes, std::ifstream* is,
              std::ofstream* os, unsigned long long* output_bytes,
              Predictor* p);

void Decompress(unsigned long long output_length, std::ifstream* is,
                std::ofstream* os, Predictor* p);

bool RunCompression(const std::string& input_path,
                    const std::string& output_path,
                    unsigned long long* input_bytes,
                    unsigned long long* output_bytes);

bool RunDecompression(const std::string& input_path,
                      const std::string& output_path,
                      unsigned long long* input_bytes,
                      unsigned long long* output_bytes);

bool RunGeneration(const std::string& input_path,
                   const std::string& output_path, int output_size);

}  // namespace runner_utils