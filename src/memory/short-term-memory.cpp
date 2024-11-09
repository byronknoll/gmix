#include "short-term-memory.h"

void ShortTermMemory::WriteToDisk(std::ofstream* s) {
  SerializeArray(s, predictions);
  Serialize(s, new_bit);
  Serialize(s, recent_bits);
  Serialize(s, bit_context);
  Serialize(s, last_byte);
  Serialize(s, always_zero);
  Serialize(s, last_two_bytes_context);
  Serialize(s, last_three_bytes_context);
  Serialize(s, last_three_bytes_15_bit_hash);
  Serialize(s, last_three_bytes_16_bit_hash);
  Serialize(s, last_four_bytes_context);
  Serialize(s, last_four_bytes_15_bit_hash);
  Serialize(s, last_four_bytes_21_bit_hash);
  Serialize(s, last_five_bytes_context);
  Serialize(s, last_five_bytes_15_bit_hash);
  Serialize(s, last_five_bytes_21_bit_hash);
  Serialize(s, last_six_bytes_context);
  Serialize(s, last_six_bytes_15_bit_hash);
  Serialize(s, last_six_bytes_21_bit_hash);
  Serialize(s, indirect_1_8_1_8);
  Serialize(s, indirect_1_8_2_16);
  Serialize(s, indirect_1_8_3_15);
  Serialize(s, indirect_2_16_1_8);
  Serialize(s, indirect_2_16_2_16);
  Serialize(s, indirect_2_16_3_15);
  Serialize(s, indirect_3_24_1_8);
  Serialize(s, indirect_4_24_2_16);
  Serialize(s, indirect_4_24_3_15);
  Serialize(s, interval_16_4);
  Serialize(s, interval_16_8);
  Serialize(s, interval_16_12);
  Serialize(s, interval_32_3);
  Serialize(s, interval_32_6);
  Serialize(s, interval_32_9);
  Serialize(s, interval_32_12);
  Serialize(s, interval_64_4);
  Serialize(s, interval_64_8);
  Serialize(s, interval_64_12);
  Serialize(s, second_last_byte);
  Serialize(s, third_last_byte);
  Serialize(s, fourth_last_byte);
  Serialize(s, fifth_last_byte);
  Serialize(s, sixth_last_byte);
  Serialize(s, seventh_last_byte);
  Serialize(s, eighth_last_byte);
  Serialize(s, ninth_last_byte);
  Serialize(s, tenth_last_byte);
  Serialize(s, last_byte_plus_recent);
  Serialize(s, second_last_plus_recent);
  SerializeArray(s, mixer_layer0_outputs);
  SerializeArray(s, mixer_layer1_outputs);
  Serialize(s, final_mixer_output);
  Serialize(s, longest_match);
  Serialize(s, bits_seen);
  SerializeArray(s, entropy);
  Serialize(s, lstm_prediction_context);
}

void ShortTermMemory::ReadFromDisk(std::ifstream* s) {
  SerializeArray(s, predictions);
  Serialize(s, new_bit);
  Serialize(s, recent_bits);
  Serialize(s, bit_context);
  Serialize(s, last_byte);
  Serialize(s, always_zero);
  Serialize(s, last_two_bytes_context);
  Serialize(s, last_three_bytes_context);
  Serialize(s, last_three_bytes_15_bit_hash);
  Serialize(s, last_three_bytes_16_bit_hash);
  Serialize(s, last_four_bytes_context);
  Serialize(s, last_four_bytes_15_bit_hash);
  Serialize(s, last_four_bytes_21_bit_hash);
  Serialize(s, last_five_bytes_context);
  Serialize(s, last_five_bytes_15_bit_hash);
  Serialize(s, last_five_bytes_21_bit_hash);
  Serialize(s, last_six_bytes_context);
  Serialize(s, last_six_bytes_15_bit_hash);
  Serialize(s, last_six_bytes_21_bit_hash);
  Serialize(s, indirect_1_8_1_8);
  Serialize(s, indirect_1_8_2_16);
  Serialize(s, indirect_1_8_3_15);
  Serialize(s, indirect_2_16_1_8);
  Serialize(s, indirect_2_16_2_16);
  Serialize(s, indirect_2_16_3_15);
  Serialize(s, indirect_3_24_1_8);
  Serialize(s, indirect_4_24_2_16);
  Serialize(s, indirect_4_24_3_15);
  Serialize(s, interval_16_4);
  Serialize(s, interval_16_8);
  Serialize(s, interval_16_12);
  Serialize(s, interval_32_3);
  Serialize(s, interval_32_6);
  Serialize(s, interval_32_9);
  Serialize(s, interval_32_12);
  Serialize(s, interval_64_4);
  Serialize(s, interval_64_8);
  Serialize(s, interval_64_12);
  Serialize(s, second_last_byte);
  Serialize(s, third_last_byte);
  Serialize(s, fourth_last_byte);
  Serialize(s, fifth_last_byte);
  Serialize(s, sixth_last_byte);
  Serialize(s, seventh_last_byte);
  Serialize(s, eighth_last_byte);
  Serialize(s, ninth_last_byte);
  Serialize(s, tenth_last_byte);
  Serialize(s, last_byte_plus_recent);
  Serialize(s, second_last_plus_recent);
  SerializeArray(s, mixer_layer0_outputs);
  SerializeArray(s, mixer_layer1_outputs);
  Serialize(s, final_mixer_output);
  Serialize(s, longest_match);
  Serialize(s, bits_seen);
  SerializeArray(s, entropy);
  Serialize(s, lstm_prediction_context);
}

void ShortTermMemory::Copy(const MemoryInterface* m) {
  const ShortTermMemory* orig = static_cast<const ShortTermMemory*>(m);
  predictions = orig->predictions;
  new_bit = orig->new_bit;
  recent_bits = orig->recent_bits;
  bit_context = orig->bit_context;
  last_byte = orig->last_byte;
  always_zero = orig->always_zero;
  last_two_bytes_context = orig->last_two_bytes_context;
  last_three_bytes_context = orig->last_three_bytes_context;
  last_three_bytes_15_bit_hash = orig->last_three_bytes_15_bit_hash;
  last_three_bytes_16_bit_hash = orig->last_three_bytes_16_bit_hash;
  last_four_bytes_context = orig->last_four_bytes_context;
  last_four_bytes_15_bit_hash = orig->last_four_bytes_15_bit_hash;
  last_four_bytes_21_bit_hash = orig->last_four_bytes_21_bit_hash;
  last_five_bytes_context = orig->last_five_bytes_context;
  last_five_bytes_15_bit_hash = orig->last_five_bytes_15_bit_hash;
  last_five_bytes_21_bit_hash = orig->last_five_bytes_21_bit_hash;
  last_six_bytes_context = orig->last_six_bytes_context;
  last_six_bytes_15_bit_hash = orig->last_six_bytes_15_bit_hash;
  last_six_bytes_21_bit_hash = orig->last_six_bytes_21_bit_hash;
  indirect_1_8_1_8 = orig->indirect_1_8_1_8;
  indirect_1_8_2_16 = orig->indirect_1_8_2_16;
  indirect_1_8_3_15 = orig->indirect_1_8_3_15;
  indirect_2_16_1_8 = orig->indirect_2_16_1_8;
  indirect_2_16_2_16 = orig->indirect_2_16_2_16;
  indirect_2_16_3_15 = orig->indirect_2_16_3_15;
  indirect_3_24_1_8 = orig->indirect_3_24_1_8;
  indirect_4_24_2_16 = orig->indirect_4_24_2_16;
  indirect_4_24_3_15 = orig->indirect_4_24_3_15;
  interval_16_4 = orig->interval_16_4;
  interval_16_8 = orig->interval_16_8;
  interval_16_12 = orig->interval_16_12;
  interval_32_3 = orig->interval_32_3;
  interval_32_6 = orig->interval_32_6;
  interval_32_9 = orig->interval_32_9;
  interval_32_12 = orig->interval_32_12;
  interval_64_4 = orig->interval_64_4;
  interval_64_8 = orig->interval_64_8;
  interval_64_12 = orig->interval_64_12;
  second_last_byte = orig->second_last_byte;
  third_last_byte = orig->third_last_byte;
  fourth_last_byte = orig->fourth_last_byte;
  fifth_last_byte = orig->fifth_last_byte;
  sixth_last_byte = orig->sixth_last_byte;
  seventh_last_byte = orig->seventh_last_byte;
  eighth_last_byte = orig->eighth_last_byte;
  ninth_last_byte = orig->ninth_last_byte;
  tenth_last_byte = orig->tenth_last_byte;
  last_byte_plus_recent = orig->last_byte_plus_recent;
  second_last_plus_recent = orig->second_last_plus_recent;
  mixer_layer0_outputs = orig->mixer_layer0_outputs;
  mixer_layer1_outputs = orig->mixer_layer1_outputs;
  final_mixer_output = orig->final_mixer_output;
  longest_match = orig->longest_match;
  bits_seen = orig->bits_seen;
  entropy = orig->entropy;
  lstm_prediction_context = orig->lstm_prediction_context;
}

int ShortTermMemory::AddPrediction(std::string description,
                                   bool enable_analysis, Model* ptr) {
  ++num_predictions;
  model_descriptions.push_back(description);
  model_enable_analysis.push_back(enable_analysis);
  prediction_index_to_model_ptr[num_predictions - 1] = ptr;
  return num_predictions - 1;
}

void ShortTermMemory::SetPrediction(float prediction, int index) {
  predictions[index] = Sigmoid::Logit(prediction);
  if (prediction == 0.5) return;
  active_models.push_back(index);
}

void ShortTermMemory::SetLogitPrediction(float prediction, int index) {
  predictions[index] = prediction;
  if (prediction == 0) return;
  active_models.push_back(index);
}

int ShortTermMemory::AddMixer(std::string description, int layer_number,
                              bool enable_analysis, Model* ptr) {
  int index = 0;
  if (layer_number == 0) {
    index = num_layer0_mixers++;
  } else if (layer_number == 1) {
    index = num_layer1_mixers++;
  } else {
    index = num_layer0_mixers + num_layer1_mixers + 1;
  }
  model_descriptions.push_back(description);
  model_enable_analysis.push_back(enable_analysis);
  mixer_index_to_model_ptr.push_back(ptr);
  return index;
}