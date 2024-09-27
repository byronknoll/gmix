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
  Serialize(s, last_three_bytes_context);
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
  Serialize(s, last_three_bytes_context);
  SerializeArray(s, mixer_outputs);
  Serialize(s, final_mixer_output);
}

void ShortTermMemory::Copy(const MemoryInterface* m) {
  const ShortTermMemory* orig = static_cast<const ShortTermMemory*>(m);
  predictions = orig->predictions;
  new_bit = orig->new_bit;
  recent_bits = orig->recent_bits;
  bit_context = orig->bit_context;
  last_byte = orig->last_byte;
  always_zero = orig->always_zero;
  last_byte_context = orig->last_byte_context;
  last_two_bytes_context = orig->last_two_bytes_context;
  last_three_bytes_context = orig->last_three_bytes_context;
  mixer_outputs = orig->mixer_outputs;
  final_mixer_output = orig->final_mixer_output;
}