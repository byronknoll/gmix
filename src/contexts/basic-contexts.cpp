#include "basic-contexts.h"

#include "murmur-hash.h"

void BasicContexts::Predict(ShortTermMemory& short_term_memory,
                            const LongTermMemory& long_term_memory) {
  if (first_prediction_) {
    // Don't update state on the very first prediction.
    first_prediction_ = false;
    return;
  }
  short_term_memory.recent_bits +=
      short_term_memory.recent_bits + short_term_memory.new_bit;
  if (short_term_memory.recent_bits >= 256) {
    short_term_memory.last_byte = short_term_memory.recent_bits - 256;
    short_term_memory.recent_bits = 1;
    short_term_memory.last_byte_context = short_term_memory.last_byte;
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
    MurmurHash3_x86_32(&short_term_memory.last_three_bytes_context, 4,
                       0XDEADBEEF, &hash);
    short_term_memory.last_three_bytes_15_bit_hash = hash & 0x7FFF;
    MurmurHash3_x86_32(&short_term_memory.last_four_bytes_context, 4,
                       0XDEADBEEF, &hash);
    short_term_memory.last_four_bytes_15_bit_hash = hash & 0x7FFF;
    MurmurHash3_x86_32(&short_term_memory.last_five_bytes_context, 8,
                       0XDEADBEEF, &hash);
    short_term_memory.last_five_bytes_15_bit_hash = hash & 0x7FFF;
  }
  short_term_memory.bit_context = short_term_memory.recent_bits - 1;
  short_term_memory.longest_match = 0;
}

void BasicContexts::Learn(const ShortTermMemory& short_term_memory,
                          LongTermMemory& long_term_memory) {
  int current_byte =
      short_term_memory.recent_bits * 2 + short_term_memory.new_bit;
  if (current_byte >= 256) {
    // A new byte has been observed.
    long_term_memory.history.push_back(current_byte);
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