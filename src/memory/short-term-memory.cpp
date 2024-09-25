#include "short-term-memory.h"

void ShortTermMemory::WriteToDisk(std::ofstream* s) {
  SerializeArray(s, predictions);
  Serialize(s, new_bit);
  Serialize(s, recent_bits);
  Serialize(s, bit_context);
  Serialize(s, last_byte);
  Serialize(s, always_zero);
  Serialize(s, last_byte_context);
  Serialize(s, last_two_bytes_context);
  SerializeArray(s, mixer_outputs);
  Serialize(s, final_mixer_output);
}

void ShortTermMemory::ReadFromDisk(std::ifstream* s) {
  SerializeArray(s, predictions);
  Serialize(s, new_bit);
  Serialize(s, recent_bits);
  Serialize(s, bit_context);
  Serialize(s, last_byte);
  Serialize(s, always_zero);
  Serialize(s, last_byte_context);
  Serialize(s, last_two_bytes_context);
  SerializeArray(s, mixer_outputs);
  Serialize(s, final_mixer_output);
}