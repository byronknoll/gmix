#include "basic-contexts.h"

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
        ((short_term_memory.last_two_bytes_context % 256) << 8) +
        short_term_memory.last_byte;
  }
  short_term_memory.bit_context = short_term_memory.recent_bits - 1;
}

void BasicContexts::WriteToDisk(std::ofstream* os) {
  os->write(reinterpret_cast<char*>(&first_prediction_), sizeof(first_prediction_));
}

void BasicContexts::ReadFromDisk(std::ifstream* is) {
  is->read(reinterpret_cast<char*>(&first_prediction_), sizeof(first_prediction_));
}