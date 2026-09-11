#include "word-contexts.h"

#include <algorithm>

WordContexts::WordContexts() {
  last_pos_.fill(-1);
  following_variety_.fill(0);
  preceding_variety_.fill(0);
  following_bits_.fill(0);
  preceding_bits_.fill(0);
  separator_score_.fill(0.0f);
}

void WordContexts::Predict(ShortTermMemory& short_term_memory,
                           const LongTermMemory& long_term_memory) {
  if (short_term_memory.recent_bits == 1 && short_term_memory.bits_seen >= 8) {
    unsigned char c = short_term_memory.last_byte;
    if (prev_byte_ >= 0) {
      uint32_t pair = (static_cast<uint32_t>(prev_byte_) << 8) | c;
      if ((following_bits_[pair >> 6] & (1ULL << (pair & 63))) == 0) {
        following_bits_[pair >> 6] |= (1ULL << (pair & 63));
        ++following_variety_[prev_byte_];
      }
      uint32_t rpair = (static_cast<uint32_t>(c) << 8) | prev_byte_;
      if ((preceding_bits_[rpair >> 6] & (1ULL << (rpair & 63))) == 0) {
        preceding_bits_[rpair >> 6] |= (1ULL << (rpair & 63));
        ++preceding_variety_[c];
      }
    }
    prev_byte_ = c;

    int64_t d = (last_pos_[c] >= 0) ? (byte_pos_ - last_pos_[c]) : 999999;
    last_pos_[c] = byte_pos_;
    ++byte_pos_;

    if (d >= 2 && d <= 32) {
      int var = std::min(following_variety_[c], preceding_variety_[c]);
      float reward = 1.0f + var * 0.1f;
      separator_score_[c] = separator_score_[c] * 0.998f + reward;
    } else {
      separator_score_[c] = separator_score_[c] * 0.998f - 0.5f;
      if (separator_score_[c] < 0.0f) separator_score_[c] = 0.0f;
    }

    if ((byte_pos_ & 31) == 0) {
      int best = 0;
      int second_best = -1;
      float max_s = separator_score_[0];
      float second_max_s = 0.0f;
      for (int i = 1; i < 256; ++i) {
        if (separator_score_[i] > max_s) {
          second_max_s = max_s;
          second_best = best;
          max_s = separator_score_[i];
          best = i;
        } else if (separator_score_[i] > second_max_s) {
          second_max_s = separator_score_[i];
          second_best = i;
        }
      }
      if (max_s > 15.0f) {
        if (detected_separator_ < 0 ||
            max_s > 1.25f * separator_score_[detected_separator_]) {
          detected_separator_ = best;
        }
      }
      if (second_max_s > 10.0f) {
        detected_secondary_delimiter_ = second_best;
      }
    }

    // Delimiter distances
    int64_t d1 = (detected_separator_ >= 0 && last_pos_[detected_separator_] >= 0)
                     ? (byte_pos_ - last_pos_[detected_separator_])
                     : 255;
    int64_t d2 = (detected_secondary_delimiter_ >= 0 &&
                  last_pos_[detected_secondary_delimiter_] >= 0)
                     ? (byte_pos_ - last_pos_[detected_secondary_delimiter_])
                     : 255;
    unsigned int dist1 = std::min<int64_t>(d1, 255);
    unsigned int dist2 = std::min<int64_t>(d2, 255);

    // Word tracking based on dynamically detected separator
    if (detected_separator_ >= 0 &&
        c == static_cast<unsigned char>(detected_separator_)) {
      if (cur_word_len_ > 0) {
        word_5_ = word_4_;
        word_4_ = word_3_;
        word_3_ = word_2_;
        word_2_ = word_1_;
        word_1_ = cur_word_;
        prev_word_len_ = cur_word_len_;
        cur_word_ = 0;
        cur_word_len_ = 0;
        first_byte_ = 0;
      }
    } else {
      if (cur_word_len_ == 0) {
        first_byte_ = c;
      }
      cur_word_ = cur_word_ * 8416 + c;
      ++cur_word_len_;
    }

    // Assign to short_term_memory
    short_term_memory.word_0 = cur_word_;
    short_term_memory.word_1 = word_1_;
    short_term_memory.word_2 = word_2_;
    short_term_memory.word_3 = word_3_;
    short_term_memory.first_byte = first_byte_;
    short_term_memory.word_len = std::min(cur_word_len_, 15);
    short_term_memory.word_len_1 =
        (std::min(prev_word_len_, 15) << 4) | std::min(cur_word_len_, 15);
    short_term_memory.first_byte_1 = (word_1_ * 263) ^ first_byte_;
    short_term_memory.word_prefix_plus_byte = (cur_word_ * 271) ^ c;

    short_term_memory.dist_to_delim1 = dist1;
    short_term_memory.dist_to_delim2 = dist2;
    short_term_memory.dist_delim1_byte = (std::min(dist1, 63u) << 8) | c;
    short_term_memory.dist_delim2_byte = (std::min(dist2, 63u) << 8) | c;

    // Word combination contexts:
    uint32_t h01 = word_1_ * 997 * 16 + cur_word_;
    short_term_memory.word_0_1 = h01 ^ (h01 >> 16);

    uint32_t h02 = word_2_ * 997 * 16 + cur_word_;
    short_term_memory.word_0_2 = h02 ^ (h02 >> 16);

    uint32_t h03 = word_3_ * 997 * 16 + cur_word_;
    short_term_memory.word_0_3 = h03 ^ (h03 >> 16);

    uint32_t h012 = (word_2_ * 29 * 31) + (word_1_ * 997 * 16) + cur_word_;
    short_term_memory.word_0_1_2 = h012 ^ (h012 >> 16);

    uint32_t h013 = (word_3_ * 29 * 31) + (word_1_ * 997 * 16) + cur_word_;
    short_term_memory.word_0_1_3 = h013 ^ (h013 >> 16);

    uint32_t h023 = (word_3_ * 29 * 31) + (word_2_ * 997 * 16) + cur_word_;
    short_term_memory.word_0_2_3 = h023 ^ (h023 >> 16);

    uint32_t h0123 = (word_3_ * 29 * 31 * 37) + (word_2_ * 29 * 31) +
                     (word_1_ * 997 * 16) + cur_word_;
    short_term_memory.word_0_1_2_3 = h0123 ^ (h0123 >> 16);

    uint32_t h12 = word_2_ * 997 * 16 + word_1_;
    short_term_memory.word_1_2 = h12 ^ (h12 >> 16);

    uint32_t h13 = word_3_ * 997 * 16 + word_1_;
    short_term_memory.word_1_3 = h13 ^ (h13 >> 16);

    uint32_t h14 = word_4_ * 997 * 16 + word_1_;
    short_term_memory.word_1_4 = h14 ^ (h14 >> 16);

    uint32_t h23 = word_3_ * 997 * 16 + word_2_;
    short_term_memory.word_2_3 = h23 ^ (h23 >> 16);

    uint32_t h34 = word_4_ * 997 * 16 + word_3_;
    short_term_memory.word_3_4 = h34 ^ (h34 >> 16);

    uint32_t h123 = (word_3_ * 29 * 31 * 37) + (word_2_ * 29 * 31) + word_1_;
    short_term_memory.word_1_2_3 = h123 ^ (h123 >> 16);

    uint32_t h124 = (word_4_ * 29 * 31 * 37) + (word_2_ * 29 * 31) + word_1_;
    short_term_memory.word_1_2_4 = h124 ^ (h124 >> 16);

    uint32_t h1234 = (word_4_ * 29 * 31 * 37 * 41) +
                     (word_3_ * 29 * 31 * 37) + (word_2_ * 29 * 31) + word_1_;
    short_term_memory.word_1_2_3_4 = h1234 ^ (h1234 >> 16);
  }
}

void WordContexts::WriteToDisk(std::ofstream* s) {
  SerializeArray(s, last_pos_);
  SerializeArray(s, following_variety_);
  SerializeArray(s, preceding_variety_);
  SerializeArray(s, following_bits_);
  SerializeArray(s, preceding_bits_);
  SerializeArray(s, separator_score_);
  Serialize(s, detected_separator_);
  Serialize(s, detected_secondary_delimiter_);
  Serialize(s, cur_word_);
  Serialize(s, cur_word_len_);
  Serialize(s, prev_word_len_);
  Serialize(s, first_byte_);
  Serialize(s, word_1_);
  Serialize(s, word_2_);
  Serialize(s, word_3_);
  Serialize(s, word_4_);
  Serialize(s, word_5_);
  Serialize(s, prev_byte_);
  Serialize(s, byte_pos_);
}

void WordContexts::ReadFromDisk(std::ifstream* s) {
  SerializeArray(s, last_pos_);
  SerializeArray(s, following_variety_);
  SerializeArray(s, preceding_variety_);
  SerializeArray(s, following_bits_);
  SerializeArray(s, preceding_bits_);
  SerializeArray(s, separator_score_);
  Serialize(s, detected_separator_);
  Serialize(s, detected_secondary_delimiter_);
  Serialize(s, cur_word_);
  Serialize(s, cur_word_len_);
  Serialize(s, prev_word_len_);
  Serialize(s, first_byte_);
  Serialize(s, word_1_);
  Serialize(s, word_2_);
  Serialize(s, word_3_);
  Serialize(s, word_4_);
  Serialize(s, word_5_);
  Serialize(s, prev_byte_);
  Serialize(s, byte_pos_);
}

void WordContexts::Copy(const MemoryInterface* m) {
  const WordContexts* orig = static_cast<const WordContexts*>(m);
  last_pos_ = orig->last_pos_;
  following_variety_ = orig->following_variety_;
  preceding_variety_ = orig->preceding_variety_;
  following_bits_ = orig->following_bits_;
  preceding_bits_ = orig->preceding_bits_;
  separator_score_ = orig->separator_score_;
  detected_separator_ = orig->detected_separator_;
  detected_secondary_delimiter_ = orig->detected_secondary_delimiter_;
  cur_word_ = orig->cur_word_;
  cur_word_len_ = orig->cur_word_len_;
  prev_word_len_ = orig->prev_word_len_;
  first_byte_ = orig->first_byte_;
  word_1_ = orig->word_1_;
  word_2_ = orig->word_2_;
  word_3_ = orig->word_3_;
  word_4_ = orig->word_4_;
  word_5_ = orig->word_5_;
  prev_byte_ = orig->prev_byte_;
  byte_pos_ = orig->byte_pos_;
}

unsigned long long WordContexts::GetMemoryUsage(
    const ShortTermMemory& short_term_memory,
    const LongTermMemory& long_term_memory) {
  return sizeof(*this);
}
