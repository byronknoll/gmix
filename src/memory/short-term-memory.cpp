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
  Serialize(s, last_three_bytes_15_bit_hash);
  Serialize(s, last_four_bytes_context);
  Serialize(s, last_four_bytes_15_bit_hash);
  Serialize(s, last_five_bytes_context);
  Serialize(s, last_five_bytes_15_bit_hash);
  Serialize(s, last_five_bytes_21_bit_hash);
  Serialize(s, indirect_1_8_1_8);
  Serialize(s, indirect_1_8_2_16);
  Serialize(s, indirect_1_8_3_15);
  Serialize(s, indirect_2_16_1_8);
  Serialize(s, indirect_2_16_2_16);
  Serialize(s, indirect_3_24_1_8);
  Serialize(s, indirect_4_24_2_15);
  Serialize(s, second_last_byte);
  SerializeArray(s, mixer_outputs);
  Serialize(s, final_mixer_output);
  Serialize(s, longest_match);
  Serialize(s, bits_seen);
  SerializeArray(s, entropy);
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
  Serialize(s, last_three_bytes_15_bit_hash);
  Serialize(s, last_four_bytes_context);
  Serialize(s, last_four_bytes_15_bit_hash);
  Serialize(s, last_five_bytes_context);
  Serialize(s, last_five_bytes_15_bit_hash);
  Serialize(s, last_five_bytes_21_bit_hash);
  Serialize(s, indirect_1_8_1_8);
  Serialize(s, indirect_1_8_2_16);
  Serialize(s, indirect_1_8_3_15);
  Serialize(s, indirect_2_16_1_8);
  Serialize(s, indirect_2_16_2_16);
  Serialize(s, indirect_3_24_1_8);
  Serialize(s, indirect_4_24_2_15);
  Serialize(s, second_last_byte);
  SerializeArray(s, mixer_outputs);
  Serialize(s, final_mixer_output);
  Serialize(s, longest_match);
  Serialize(s, bits_seen);
  SerializeArray(s, entropy);
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
  last_three_bytes_15_bit_hash = orig->last_three_bytes_15_bit_hash;
  last_four_bytes_context = orig->last_four_bytes_context;
  last_four_bytes_15_bit_hash = orig->last_four_bytes_15_bit_hash;
  last_five_bytes_context = orig->last_five_bytes_context;
  last_five_bytes_15_bit_hash = orig->last_five_bytes_15_bit_hash;
  last_five_bytes_21_bit_hash = orig->last_five_bytes_21_bit_hash;
  indirect_1_8_1_8 = orig->indirect_1_8_1_8;
  indirect_1_8_2_16 = orig->indirect_1_8_2_16;
  indirect_1_8_3_15 = orig->indirect_1_8_3_15;
  indirect_2_16_1_8 = orig->indirect_2_16_1_8;
  indirect_2_16_2_16 = orig->indirect_2_16_2_16;
  indirect_3_24_1_8 = orig->indirect_3_24_1_8;
  indirect_4_24_2_15 = orig->indirect_4_24_2_15;
  second_last_byte = orig->second_last_byte;
  mixer_outputs = orig->mixer_outputs;
  final_mixer_output = orig->final_mixer_output;
  longest_match = orig->longest_match;
  bits_seen = orig->bits_seen;
  entropy = orig->entropy;
}