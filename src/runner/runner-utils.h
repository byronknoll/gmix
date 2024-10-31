#include <string.h>

#include <fstream>

#include "../predictor.h"

namespace runner_utils {
float Rand();

void WriteHeader(unsigned long long length, std::ofstream* os);

void ReadHeader(std::ifstream* is, unsigned long long* length);

void ClearOutput();

void Compress(unsigned long long input_bytes, std::ifstream* is,
              std::ofstream* os, unsigned long long* output_bytes,
              Predictor* p);

void Decompress(unsigned long long output_length, std::ifstream* is,
                std::ofstream* os, Predictor* p);

// checkpoint_path can be empty (when not using a pretrained model).
bool RunCompression(const std::string& checkpoint_path,
                    const std::string& input_path,
                    const std::string& output_path,
                    unsigned long long* input_bytes,
                    unsigned long long* output_bytes);

// checkpoint_path can be empty (when not using a pretrained model).
bool RunDecompression(const std::string& checkpoint_path,
                      const std::string& input_path,
                      const std::string& output_path,
                      unsigned long long* input_bytes,
                      unsigned long long* output_bytes);

bool RunGeneration(const std::string& checkpoint_path,
                   const std::string& prompt_path,
                   const std::string& output_path, int output_size,
                   float temperature);

// checkpoint_path can be empty (when not using a pretrained model).
bool RunTraining(const std::string& checkpoint_path,
                 const std::string& train_path, const std::string& test_path,
                 unsigned long long* input_bytes,
                 unsigned long long* output_bytes);

}  // namespace runner_utils