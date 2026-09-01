#include "indirect-hash.h"

#include "murmur-hash.h"

IndirectHash::IndirectHash(int outer_order, unsigned int table_size,
                           int inner_order, unsigned int& output_context)
    : table_(table_size, 0),
      outer_order_(outer_order),
      inner_mod_(1ULL << (8 * (inner_order - 1))),
      context_(output_context) {
  table_.shrink_to_fit();
}

void IndirectHash::Predict(ShortTermMemory& short_term_memory,
                           const LongTermMemory& long_term_memory) {
  if (short_term_memory.recent_bits == 1) {  // Byte boundary
    // Update the previous outer context's inner context with the last observed byte.
    unsigned int idx = outer_hash_ % table_.size();
    unsigned int& inner_context = table_[idx];
    inner_context =
        ((inner_context % inner_mod_) << 8) | short_term_memory.last_byte;

    // Direct, zero-hash outer key generation:
    if (outer_order_ == 1) {
      outer_hash_ = short_term_memory.last_byte;
    } else if (outer_order_ == 2) {
      outer_hash_ = (static_cast<unsigned int>(short_term_memory.recent_bytes[1]) << 8) |
                    short_term_memory.last_byte;
    } else if (outer_order_ == 3) {
      outer_hash_ = short_term_memory.last_three_bytes_hash;
    } else if (outer_order_ == 4) {
      outer_hash_ = short_term_memory.last_four_bytes_hash;
    } else {
      outer_hash_ = short_term_memory.last_six_bytes_hash;
    }

    // Map to the new context by hashing inner context to ensure uniform table distribution.
    MurmurHash3_x86_32(&(table_[outer_hash_ % table_.size()]), 4, 0XDEADBEEF,
                       &context_);
  }
}

void IndirectHash::WriteToDisk(std::ofstream* s) {
  std::vector<unsigned int> keys;
  for (size_t i = 0; i < table_.size(); ++i) {
    if (table_[i] != 0) {
      keys.push_back(i);
    }
  }
  unsigned int size = keys.size();
  Serialize(s, size);
  if (size < table_.size() / 2) {
    // If the table is sparse, encode keys+values.
    for (unsigned int key : keys) {
      Serialize(s, key);
      Serialize(s, table_[key]);
    }
  } else {
    // If the table is dense, encode all values.
    SerializeArray(s, table_);
  }
  Serialize(s, outer_hash_);
}

void IndirectHash::ReadFromDisk(std::ifstream* s) {
  unsigned int size;
  Serialize(s, size);
  if (size < table_.size() / 2) {
    // If the table is sparse, encode keys+values.
    for (unsigned int i = 0; i < size; ++i) {
      unsigned int key;
      Serialize(s, key);
      unsigned int context;
      Serialize(s, context);
      table_[key] = context;
    }
  } else {
    // If the table is dense, encode all values.
    SerializeArray(s, table_);
  }
  Serialize(s, outer_hash_);
}

void IndirectHash::Copy(const MemoryInterface* m) {
  const IndirectHash* orig = static_cast<const IndirectHash*>(m);
  table_ = orig->table_;
  outer_hash_ = orig->outer_hash_;
}

unsigned long long IndirectHash::GetMemoryUsage(
    const ShortTermMemory& short_term_memory,
    const LongTermMemory& long_term_memory) {
  unsigned long long usage = sizeof(*this);
  usage += table_.size() * sizeof(unsigned int);
  return usage;
}