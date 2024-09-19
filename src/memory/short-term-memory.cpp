#include "short-term-memory.h"

void ShortTermMemory::WriteToDisk(std::ofstream* os) {
  for (int i = 0; i < predictions.size(); ++i) {
    os->write(reinterpret_cast<char*>(&predictions[i]), sizeof(predictions[i]));
  }
  os->write(reinterpret_cast<char*>(&new_bit), sizeof(new_bit));
  os->write(reinterpret_cast<char*>(&recent_bits), sizeof(recent_bits));
  os->write(reinterpret_cast<char*>(&bit_context), sizeof(bit_context));
  os->write(reinterpret_cast<char*>(&last_byte), sizeof(last_byte));
  os->write(reinterpret_cast<char*>(&always_zero), sizeof(always_zero));
  os->write(reinterpret_cast<char*>(&last_byte_context),
            sizeof(last_byte_context));
  os->write(reinterpret_cast<char*>(&last_two_bytes_context),
            sizeof(last_two_bytes_context));
  for (int i = 0; i < mixer_outputs.size(); ++i) {
    os->write(reinterpret_cast<char*>(&mixer_outputs[i]),
              sizeof(mixer_outputs[i]));
  }
  os->write(reinterpret_cast<char*>(&final_mixer_output),
            sizeof(final_mixer_output));
}

void ShortTermMemory::ReadFromDisk(std::ifstream* is) {
  for (int i = 0; i < predictions.size(); ++i) {
    is->read(reinterpret_cast<char*>(&predictions[i]), sizeof(predictions[i]));
  }
  is->read(reinterpret_cast<char*>(&new_bit), sizeof(new_bit));
  is->read(reinterpret_cast<char*>(&recent_bits), sizeof(recent_bits));
  is->read(reinterpret_cast<char*>(&bit_context), sizeof(bit_context));
  is->read(reinterpret_cast<char*>(&last_byte), sizeof(last_byte));
  is->read(reinterpret_cast<char*>(&always_zero), sizeof(always_zero));
  is->read(reinterpret_cast<char*>(&last_byte_context),
           sizeof(last_byte_context));
  is->read(reinterpret_cast<char*>(&last_two_bytes_context),
           sizeof(last_two_bytes_context));
  for (int i = 0; i < mixer_outputs.size(); ++i) {
    is->read(reinterpret_cast<char*>(&mixer_outputs[i]),
             sizeof(mixer_outputs[i]));
  }
  is->read(reinterpret_cast<char*>(&final_mixer_output),
           sizeof(final_mixer_output));
}