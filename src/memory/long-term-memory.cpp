#include "long-term-memory.h"

#include <cstring>
#include <set>

void LongTermMemory::WriteToDisk(std::ofstream* s) {
  for (auto& mem_ptr : direct) {
    std::set<unsigned int> keys;  // use a set to get consistent key order.
    for (auto& it : mem_ptr->predictions) {
      keys.insert(it.first);
    }
    unsigned int size = keys.size();
    Serialize(s, size);
    for (unsigned int key : keys) {
      Serialize(s, key);
      Serialize(s, mem_ptr->predictions[key].prediction);
      Serialize(s, mem_ptr->predictions[key].count);
    }
  }

  for (auto& ptr : mixers) {
    unsigned int mixer_size = ptr->mixer_map.size();
    Serialize(s, mixer_size);
    if (mixer_size == 0) continue;
    unsigned int input_size = ptr->mixer_map.begin()->second->weights.size();
    Serialize(s, input_size);
    std::set<unsigned int> keys;  // use a set to get consistent key order.
    for (auto& it : ptr->mixer_map) {
      keys.insert(it.first);
    }
    for (unsigned int context : keys) {
      Serialize(s, context);
      Serialize(s, ptr->mixer_map[context]->steps);
      SerializeArray(s, ptr->mixer_map[context]->weights);
    }
  }

  for (auto& x : lstm_output_layer) {
    for (auto& y : x) {
      SerializeArray(s, y);
    }
  }
  for (auto& x : neuron_layer_weights) {
    for (auto& y : x->weights) {
      SerializeArray(s, y);
    }
  }
  // PPM memory can have long "zero" sequences. We can reduce the serialization
  // size by storing the position/size of those.
  unsigned long long zero_sequence_count = 0;
  unsigned long long zero_sequence_start = 0;
  std::vector<unsigned long long> zero_sequence_counts;
  std::vector<unsigned long long> zero_sequence_starts;
  for (unsigned long long i = 0; i < ppmd_memory_size; ++i) {
    if (ppmd_memory[i] == 0) {
      if (zero_sequence_count == 0) {
        zero_sequence_start = i;
      }
      ++zero_sequence_count;
    } else if (zero_sequence_count > 0) {
      if (zero_sequence_count > 100) {
        zero_sequence_counts.push_back(zero_sequence_count);
        zero_sequence_starts.push_back(zero_sequence_start);
      }
      zero_sequence_count = 0;
    }
  }
  int num_sequences = zero_sequence_counts.size();
  Serialize(s, num_sequences);
  for (int i = 0; i < num_sequences; ++i) {
    Serialize(s, zero_sequence_counts[i]);
    Serialize(s, zero_sequence_starts[i]);
  }
  int sequence_pos = 0;
  for (unsigned long long i = 0; i < ppmd_memory_size; ++i) {
    if (sequence_pos < zero_sequence_counts.size() &&
        i == zero_sequence_starts[sequence_pos]) {
      i += zero_sequence_counts[sequence_pos] - 1;
      ++sequence_pos;
      continue;
    }
    Serialize(s, ppmd_memory[i]);
  }
}

void LongTermMemory::ReadFromDisk(std::ifstream* s) {
  for (auto& mem_ptr : direct) {
    unsigned int size;
    Serialize(s, size);
    for (int i = 0; i < size; ++i) {
      unsigned int key;
      Serialize(s, key);
      float p;
      Serialize(s, p);
      unsigned char count;
      Serialize(s, count);
      mem_ptr->predictions[key].prediction = p;
      mem_ptr->predictions[key].count = count;
    }
  }

  for (auto& ptr : mixers) {
    ptr->mixer_map.clear();
    unsigned int mixer_size;
    Serialize(s, mixer_size);
    unsigned int input_size;
    Serialize(s, input_size);
    for (int i = 0; i < mixer_size; ++i) {
      unsigned int context;
      Serialize(s, context);
      ptr->mixer_map[context] =
          std::unique_ptr<MixerData>(new MixerData(input_size));
      Serialize(s, ptr->mixer_map[context]->steps);
      SerializeArray(s, ptr->mixer_map[context]->weights);
    }
  }

  for (auto& x : lstm_output_layer) {
    for (auto& y : x) {
      SerializeArray(s, y);
    }
  }
  for (auto& x : neuron_layer_weights) {
    for (auto& y : x->weights) {
      SerializeArray(s, y);
    }
  }
  int num_sequences;
  Serialize(s, num_sequences);
  unsigned long long zero_sequence_count = 0;
  unsigned long long zero_sequence_start = 0;
  std::vector<unsigned long long> zero_sequence_counts;
  std::vector<unsigned long long> zero_sequence_starts;
  for (int i = 0; i < num_sequences; ++i) {
    Serialize(s, zero_sequence_count);
    Serialize(s, zero_sequence_start);
    zero_sequence_counts.push_back(zero_sequence_count);
    zero_sequence_starts.push_back(zero_sequence_start);
  }
  int sequence_pos = 0;
  for (unsigned long long i = 0; i < ppmd_memory_size; ++i) {
    if (sequence_pos < zero_sequence_counts.size() &&
        i == zero_sequence_starts[sequence_pos]) {
      memset(ppmd_memory + i, 0, zero_sequence_counts[sequence_pos]);
      i += zero_sequence_counts[sequence_pos] - 1;
      ++sequence_pos;
      continue;
    }
    Serialize(s, ppmd_memory[i]);
  }
}

void LongTermMemory::Copy(const MemoryInterface* m) {
  const LongTermMemory* orig = static_cast<const LongTermMemory*>(m);
  for (int i = 0; i < direct.size(); ++i) {
    direct[i]->predictions = orig->direct[i]->predictions;
  }

  for (int i = 0; i < mixers.size(); ++i) {
    mixers[i]->mixer_map.clear();
    for (const auto& it : orig->mixers[i]->mixer_map) {
      mixers[i]->mixer_map[it.first] =
          std::unique_ptr<MixerData>(new MixerData(it.second->weights.size()));
      mixers[i]->mixer_map[it.first]->steps = it.second->steps;
      mixers[i]->mixer_map[it.first]->weights = it.second->weights;
    }
  }

  lstm_output_layer = orig->lstm_output_layer;
  for (int i = 0; i < neuron_layer_weights.size(); ++i) {
    neuron_layer_weights[i]->weights = orig->neuron_layer_weights[i]->weights;
  }
  for (unsigned long long i = 0; i < ppmd_memory_size; ++i) {
    ppmd_memory[i] = orig->ppmd_memory[i];
  }
}