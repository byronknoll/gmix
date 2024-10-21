#include "indirect-hash.h"

#include <set>

#include "murmur-hash.h"

IndirectHash::IndirectHash(int outer_order, int outer_hash_size,
                           int inner_order, int inner_hash_size,
                           unsigned int& output_context)
    : outer_mod_(1 << (8 * (outer_order - 1))),
      inner_mod_(1 << (8 * (inner_order - 1))),
      outer_hash_mod_(1 << outer_hash_size),
      inner_hash_mod_(1 << inner_hash_size),
      context_(output_context) {}

void IndirectHash::Predict(ShortTermMemory& short_term_memory,
                           const LongTermMemory& long_term_memory) {
  if (short_term_memory.recent_bits == 1) {  // Byte boundary
    // Update the previous inner context with the last byte.
    unsigned long long& inner_context = map_[outer_hash_];
    inner_context =
        ((inner_context % inner_mod_) << 8) + short_term_memory.last_byte;
    // Update the outer context with the last byte.
    outer_context_ =
        ((outer_context_ % outer_mod_) << 8) + short_term_memory.last_byte;
    unsigned int hash;
    MurmurHash3_x86_32(&outer_context_, 8, 0XDEADBEEF, &hash);
    outer_hash_ = hash % outer_hash_mod_;
    // Map to the new inner context.
    MurmurHash3_x86_32(&(map_[outer_hash_]), 8, 0XDEADBEEF, &hash);
    context_ = hash % inner_hash_mod_;
  }
}

void IndirectHash::WriteToDisk(std::ofstream* s) {
  std::set<unsigned int> keys;  // use a set to get consistent key order.
  for (const auto& it : map_) {
    keys.insert(it.first);
  }
  int size = keys.size();
  Serialize(s, size);
  for (unsigned int key : keys) {
    Serialize(s, key);
    Serialize(s, map_[key]);
  }
  Serialize(s, outer_context_);
  Serialize(s, outer_mod_);
  Serialize(s, inner_mod_);
  Serialize(s, outer_hash_);
  Serialize(s, outer_hash_mod_);
  Serialize(s, inner_hash_mod_);
}

void IndirectHash::ReadFromDisk(std::ifstream* s) {
  map_.clear();
  int size;
  Serialize(s, size);
  for (int i = 0; i < size; ++i) {
    unsigned int key;
    Serialize(s, key);
    unsigned long long val;
    Serialize(s, val);
    map_[key] = val;
  }
  Serialize(s, outer_context_);
  Serialize(s, outer_mod_);
  Serialize(s, inner_mod_);
  Serialize(s, outer_hash_);
  Serialize(s, outer_hash_mod_);
  Serialize(s, inner_hash_mod_);
}

void IndirectHash::Copy(const MemoryInterface* m) {
  const IndirectHash* orig = static_cast<const IndirectHash*>(m);
  map_ = orig->map_;
  outer_context_ = orig->outer_context_;
  outer_mod_ = orig->outer_mod_;
  inner_mod_ = orig->inner_mod_;
  outer_hash_ = orig->outer_hash_;
  outer_hash_mod_ = orig->outer_hash_mod_;
  inner_hash_mod_ = orig->inner_hash_mod_;
}

unsigned long long IndirectHash::GetMemoryUsage(
    const ShortTermMemory& short_term_memory,
    const LongTermMemory& long_term_memory) {
  unsigned long long usage = 36;
  usage += map_.size() * 12;
  return usage;
}