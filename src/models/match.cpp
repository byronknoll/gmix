#include "match.h"

Match::Match(ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory, unsigned long long table_size,
             const unsigned int& byte_context, int limit,
             std::string description, bool enable_analysis)
    : byte_context_(byte_context),
      cur_match_(0),
      cur_byte_(0),
      bit_pos_(128),
      match_length_(0),
      limit_(limit),
      learning_rate_(1.0 / limit) {
  prediction_index_ =
      short_term_memory.AddPrediction(description, enable_analysis, this);
  memory_index_ = long_term_memory.match_memory.size();
  long_term_memory.match_memory.push_back(MatchMemory(table_size));
  auto& memory = long_term_memory.match_memory.back();
  for (int i = 0; i < 256; ++i) {
    memory.predictions[i] = 0.5 + (i + 0.5) / 512;
  }
  memory.counts.fill(1);
}

void Match::Predict(ShortTermMemory& short_term_memory,
                    const LongTermMemory& long_term_memory) {
  int match = 0;
  // Check if the current bit matches the history.
  if (short_term_memory.new_bit == ((cur_byte_ & bit_pos_) != 0)) match = 1;
  if (match) {
    if (match_length_ < 255) ++match_length_;
  } else {
    match_length_ = 0;
  }
  bit_pos_ /= 2;  // Move to the next bit.
  const auto& match_memory = long_term_memory.match_memory[memory_index_];

  if (short_term_memory.recent_bits == 1) {  // Byte boundary.
    // If the history pointer overflows, reset the match.
    if (cur_match_ == long_term_memory.history.size() - 1) {
      match_length_ = 0;
    }
    if (match_length_ < 8) {
      // There was a mismatch, so we need to find a new match.
      const auto& it =
          match_memory.table[byte_context_ % match_memory.table.size()];
      // Decode the five byte history pointer.
      cur_match_ = it[0] + (1 << 8) * it[1] + (1 << 16) * it[2] +
                   (1 << 24) * it[3] + (1ULL << 32) * it[4];
    } else {
      // The last 8 bits matched, so continue matching the next byte.
      ++cur_match_;
    }
    if (!long_term_memory.history.empty()) {
      cur_byte_ = long_term_memory.history.at(cur_match_);
    }
    bit_pos_ = 128;  // Reset the bit position.
  }

  // Only output a prediction for >2 match length.
  if (match_length_ > 2) {
    float p = 0.5;
    if (cur_byte_ & bit_pos_)
      p = match_memory.predictions[match_length_];
    else
      p = 1 - match_memory.predictions[match_length_];
    short_term_memory.SetPrediction(p, prediction_index_);
  }

  // Update the longest match context (which is used externally).
  unsigned int match_context = match_length_ / 32;
  short_term_memory.longest_match =
      std::max(short_term_memory.longest_match, match_context);
}

void Match::Learn(const ShortTermMemory& short_term_memory,
                  LongTermMemory& long_term_memory) {
  // Only update counts/predictions for >2 match length.
  if (match_length_ > 2) {
    int match = 0;
    if (short_term_memory.new_bit == ((cur_byte_ & bit_pos_) != 0)) match = 1;
    auto& match_memory = long_term_memory.match_memory[memory_index_];
    float learning_rate = learning_rate_;
    if (match_memory.counts[match_length_] < limit_) {
      ++match_memory.counts[match_length_];
      learning_rate = 1.0 / match_memory.counts[match_length_];
    }
    match_memory.predictions[match_length_] +=
        (match - match_memory.predictions[match_length_]) * learning_rate;
  }

  if (short_term_memory.recent_bits >= 128) {  // Byte boundary.
    if (short_term_memory.longest_match >= 2) {
      // When there is a long match, the byte history (in short term memory) is
      // not updated. Here we should only replace the context if the history is
      // updated.
      return;
    }
    auto& match_memory = long_term_memory.match_memory[memory_index_];
    auto& loc = match_memory.table[byte_context_ % match_memory.table.size()];
    unsigned long long pos = long_term_memory.history.size() - 1;
    // Encode the five byte history pointer.
    loc[0] = pos;
    loc[1] = pos >> 8;
    loc[2] = pos >> 16;
    loc[3] = pos >> 24;
    loc[4] = pos >> 32;
  }
}

void Match::WriteToDisk(std::ofstream* s) {
  Serialize(s, cur_match_);
  Serialize(s, cur_byte_);
  Serialize(s, bit_pos_);
  Serialize(s, match_length_);
}

void Match::ReadFromDisk(std::ifstream* s) {
  Serialize(s, cur_match_);
  Serialize(s, cur_byte_);
  Serialize(s, bit_pos_);
  Serialize(s, match_length_);
}

void Match::Copy(const MemoryInterface* m) {
  const Match* orig = static_cast<const Match*>(m);
  cur_match_ = orig->cur_match_;
  cur_byte_ = orig->cur_byte_;
  bit_pos_ = orig->bit_pos_;
  match_length_ = orig->match_length_;
}

unsigned long long Match::GetMemoryUsage(
    const ShortTermMemory& short_term_memory,
    const LongTermMemory& long_term_memory) {
  unsigned long long usage = 27;
  usage += 256 * 4;  // predictions
  usage += 256 * 4;  // counts
  usage += 5 * long_term_memory.match_memory[memory_index_].table.size();
  return usage;
}