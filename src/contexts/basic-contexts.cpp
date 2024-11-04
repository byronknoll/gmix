#include "basic-contexts.h"

#include "murmur-hash.h"

void BasicContexts::ByteUpdate(ShortTermMemory& short_term_memory,
                               const LongTermMemory& long_term_memory) {
  short_term_memory.fourth_last_byte = short_term_memory.third_last_byte;
  short_term_memory.third_last_byte = short_term_memory.second_last_byte;
  short_term_memory.second_last_byte = short_term_memory.last_byte;
  short_term_memory.last_byte = short_term_memory.recent_bits - 256;
  short_term_memory.recent_bits = 1;
  short_term_memory.last_two_bytes_context =
      ((short_term_memory.last_two_bytes_context % (1 << 8)) << 8) +
      short_term_memory.last_byte;
  short_term_memory.last_three_bytes_context =
      ((short_term_memory.last_three_bytes_context % (1 << 16)) << 8) +
      short_term_memory.last_byte;
  short_term_memory.last_four_bytes_context =
      ((short_term_memory.last_four_bytes_context % (1 << 24)) << 8) +
      short_term_memory.last_byte;
  short_term_memory.last_five_bytes_context =
      ((short_term_memory.last_five_bytes_context % (1ULL << 32)) << 8) +
      short_term_memory.last_byte;
  unsigned int hash;
  MurmurHash3_x86_32(&short_term_memory.last_three_bytes_context, 4, 0XDEADBEEF,
                     &hash);
  short_term_memory.last_three_bytes_15_bit_hash = hash % (1 << 15);
  MurmurHash3_x86_32(&short_term_memory.last_four_bytes_context, 4, 0XDEADBEEF,
                     &hash);
  short_term_memory.last_four_bytes_15_bit_hash = hash % (1 << 15);
  MurmurHash3_x86_32(&short_term_memory.last_five_bytes_context, 8, 0XDEADBEEF,
                     &hash);
  short_term_memory.last_five_bytes_15_bit_hash = hash % (1 << 15);
  short_term_memory.last_five_bytes_21_bit_hash = hash % (1 << 21);
}

void BasicContexts::Predict(ShortTermMemory& short_term_memory,
                            const LongTermMemory& long_term_memory) {
  if (first_prediction_) {
    // Don't update state on the very first prediction.
    first_prediction_ = false;
    return;
  }
  ++short_term_memory.bits_seen;
  short_term_memory.recent_bits +=
      short_term_memory.recent_bits + short_term_memory.new_bit;
  if (short_term_memory.recent_bits >= 256) {
    ByteUpdate(short_term_memory, long_term_memory);
  }
  short_term_memory.bit_context = short_term_memory.recent_bits - 1;
  short_term_memory.last_byte_plus_recent =
      (short_term_memory.last_byte << 8) + short_term_memory.bit_context;
  short_term_memory.second_last_plus_recent =
      (short_term_memory.second_last_byte << 8) + short_term_memory.bit_context;
  short_term_memory.longest_match = 0;
}

void BasicContexts::Learn(const ShortTermMemory& short_term_memory,
                          LongTermMemory& long_term_memory) {
  int current_byte =
      short_term_memory.recent_bits * 2 + short_term_memory.new_bit;
  if (current_byte >= 256) {  // A new byte has been observed.
    // Only add the new byte to the history if there isn't a long match. This
    // helps to save memory - we don't need to keep track of sequences which
    // have already occurred before.
    if (short_term_memory.longest_match < 2) {
      long_term_memory.history.push_back(current_byte);
    }
  }
}

void BasicContexts::WriteToDisk(std::ofstream* s) {
  Serialize(s, first_prediction_);
}

void BasicContexts::ReadFromDisk(std::ifstream* s) {
  Serialize(s, first_prediction_);
}

void BasicContexts::Copy(const MemoryInterface* m) {
  const BasicContexts* orig = static_cast<const BasicContexts*>(m);
  first_prediction_ = orig->first_prediction_;
}