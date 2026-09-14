#include "basic-contexts.h"

#include "murmur-hash.h"

void BasicContexts::ByteUpdate(ShortTermMemory& short_term_memory,
                              const LongTermMemory& long_term_memory) {
  short_term_memory.last_byte = short_term_memory.recent_bits - 256;
  ++short_term_memory.rotating_history_pos;
  if (short_term_memory.rotating_history_pos ==
      short_term_memory.rotating_history.size()) {
    short_term_memory.rotating_history_pos = 0;
  }
  short_term_memory.rotating_history[short_term_memory.rotating_history_pos] =
      short_term_memory.last_byte;
  for (int i = 0; i < short_term_memory.recent_bytes.size(); ++i) {
    short_term_memory.recent_bytes[i] = short_term_memory.GetRecentByte(i);
  }
  short_term_memory.direct_two_bytes =
      (short_term_memory.recent_bytes[1] << 8) | short_term_memory.last_byte;
  short_term_memory.prev_two_bytes =
      (short_term_memory.recent_bytes[2] << 8) | short_term_memory.recent_bytes[1];

  // Periodic stride detection (inspired by PAQ8 recordModel)
  unsigned int c = short_term_memory.last_byte;
  ++byte_pos_;
  int r = byte_pos_ - cpos1_[c];
  if (r > 1 && r <= 1000 && r == cpos1_[c] - cpos2_[c] &&
      r == cpos2_[c] - cpos3_[c]) {
    if (r == candidate_stride_) {
      ++candidate_count_;
      if (candidate_count_ >= 3) {
        detected_stride_ = r;
      }
    } else {
      candidate_stride_ = r;
      candidate_count_ = 1;
    }
  }
  cpos3_[c] = cpos2_[c];
  cpos2_[c] = cpos1_[c];
  cpos1_[c] = byte_pos_;

  int active_stride = (detected_stride_ >= 2 && detected_stride_ <= 1000)
                          ? detected_stride_
                          : 2;
  short_term_memory.stride_2 =
      short_term_memory.GetRecentByte(active_stride - 1);
  short_term_memory.stride_3 = short_term_memory.GetRecentByte(2);
  short_term_memory.stride_4 = short_term_memory.GetRecentByte(3);

  short_term_memory.recent_bits = 1;
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
      (short_term_memory.recent_bytes[1] << 8) + short_term_memory.bit_context;
  short_term_memory.longest_match = 0;  // This will get updated by match model.
}

void BasicContexts::Learn(const ShortTermMemory& short_term_memory,
                          LongTermMemory& long_term_memory) {
  int current_byte =
      short_term_memory.recent_bits * 2 + short_term_memory.new_bit;
  if (current_byte >= 256) {  // A new byte has been observed.
    long_term_memory.history.push_back(current_byte);
  }
}

void BasicContexts::WriteToDisk(std::ofstream* s) {
  Serialize(s, first_prediction_);
  Serialize(s, byte_pos_);
  Serialize(s, candidate_stride_);
  Serialize(s, candidate_count_);
  Serialize(s, detected_stride_);
  SerializeArray(s, cpos1_);
  SerializeArray(s, cpos2_);
  SerializeArray(s, cpos3_);
}

void BasicContexts::ReadFromDisk(std::ifstream* s) {
  Serialize(s, first_prediction_);
  Serialize(s, byte_pos_);
  Serialize(s, candidate_stride_);
  Serialize(s, candidate_count_);
  Serialize(s, detected_stride_);
  SerializeArray(s, cpos1_);
  SerializeArray(s, cpos2_);
  SerializeArray(s, cpos3_);
}

void BasicContexts::Copy(const MemoryInterface* m) {
  const BasicContexts* orig = static_cast<const BasicContexts*>(m);
  first_prediction_ = orig->first_prediction_;
  byte_pos_ = orig->byte_pos_;
  candidate_stride_ = orig->candidate_stride_;
  candidate_count_ = orig->candidate_count_;
  detected_stride_ = orig->detected_stride_;
  cpos1_ = orig->cpos1_;
  cpos2_ = orig->cpos2_;
  cpos3_ = orig->cpos3_;
}