#include "long-term-memory.h"

#include <cstring>
#include <set>

void LongTermMemory::WriteToDisk(std::ofstream* s) {
  auto start = s->tellp();
  for (auto& mem : indirect) {
    std::vector<unsigned int> keys;
    for (int i = 0; i < mem.nonstationary_table.size(); ++i) {
      if (mem.nonstationary_table[i] != 255) {
        keys.push_back(i);
      }
    }
    unsigned int size = keys.size();
    Serialize(s, size);
    if (size < mem.nonstationary_table.size() / 3) {
      // If the table is sparse, encode keys+values.
      for (unsigned int key : keys) {
        Serialize(s, key);
        Serialize(s, mem.nonstationary_table[key]);
        Serialize(s, mem.run_map_table[key]);
      }
    } else {
      // If the table is dense, encode all values.
      SerializeArray(s, mem.nonstationary_table);
      SerializeArray(s, mem.run_map_table);
    }

    SerializeArray(s, mem.nonstationary_predictions);
    SerializeArray(s, mem.run_map_predictions);
  }
  printf("\nIndirect model size: %ld\n", s->tellp() - start);

  start = s->tellp();
  for (auto& mixer : mixers) {
    unsigned int mixer_size = mixer.mixer_map.size();
    Serialize(s, mixer_size);
    if (mixer_size == 0) continue;
    unsigned int input_size = mixer.mixer_map.begin()->second->weights.size();
    Serialize(s, input_size);
    std::set<unsigned int> keys;  // use a set to get consistent key order.
    for (auto& it : mixer.mixer_map) {
      keys.insert(it.first);
    }
    for (unsigned int context : keys) {
      Serialize(s, context);
      Serialize(s, mixer.mixer_map[context]->steps);
      SerializeArray(s, mixer.mixer_map[context]->weights);
    }
  }
  printf("Mixers size: %ld\n", s->tellp() - start);

  start = s->tellp();
  for (auto& x : lstm_output_layer) {
    for (auto& y : x) {
      SerializeArray(s, y);
    }
  }
  for (auto& x : neuron_layer_weights) {
    for (auto& y : x.weights) {
      SerializeArray(s, y);
    }
  }
  printf("LSTM size: %ld\n", s->tellp() - start);

  start = s->tellp();
  unsigned long long size = history.size();
  Serialize(s, size);
  for (unsigned long long i = 0; i < size; ++i) {
    Serialize(s, history[i]);
  }
  for (auto& mem : match_memory) {
    std::vector<unsigned int> keys;
    for (int i = 0; i < mem.table.size(); ++i) {
      bool valid = false;
      for (int j = 0; j < 5; ++j) {
        if (mem.table[i][j] != 0) {
          valid = true;
          break;
        }
      }
      if (valid) {
        keys.push_back(i);
      }
    }
    unsigned int size = keys.size();
    Serialize(s, size);
    if (size < (5.0/9.0) * mem.table.size()) {
      // If the table is sparse, encode keys+values.
      for (unsigned int key : keys) {
        Serialize(s, key);
        SerializeArray(s, mem.table[key]);
      }
    } else {
      // If the table is dense, encode all values.
      for (int i = 0; i < mem.table.size(); ++i) {
        SerializeArray(s, mem.table[i]);
      }
    }
    SerializeArray(s, mem.predictions);
    SerializeArray(s, mem.counts);
  }
  printf("Match size: %ld\n", s->tellp() - start);
}

void LongTermMemory::ReadFromDisk(std::ifstream* s) {
  for (auto& mem : indirect) {
    unsigned int size;
    Serialize(s, size);
    if (size < mem.nonstationary_table.size() / 3) {
      // If the table is sparse, encode keys+values.
      for (int i = 0; i < size; ++i) {
        unsigned int key;
        Serialize(s, key);
        unsigned char state;
        Serialize(s, state);
        mem.nonstationary_table[key] = state;
        Serialize(s, state);
        mem.run_map_table[key] = state;
      }
    } else {
      // If the table is dense, encode all values.
      SerializeArray(s, mem.nonstationary_table);
      SerializeArray(s, mem.run_map_table);
    }
    SerializeArray(s, mem.nonstationary_predictions);
    SerializeArray(s, mem.run_map_predictions);
  }

  for (auto& mixer : mixers) {
    mixer.mixer_map.clear();
    unsigned int mixer_size;
    Serialize(s, mixer_size);
    unsigned int input_size;
    Serialize(s, input_size);
    for (int i = 0; i < mixer_size; ++i) {
      unsigned int context;
      Serialize(s, context);
      mixer.mixer_map[context] =
          std::unique_ptr<MixerData>(new MixerData(input_size));
      Serialize(s, mixer.mixer_map[context]->steps);
      SerializeArray(s, mixer.mixer_map[context]->weights);
    }
  }

  for (auto& x : lstm_output_layer) {
    for (auto& y : x) {
      SerializeArray(s, y);
    }
  }
  for (auto& x : neuron_layer_weights) {
    for (auto& y : x.weights) {
      SerializeArray(s, y);
    }
  }

  unsigned long long size;
  history.clear();
  Serialize(s, size);
  for (unsigned long long i = 0; i < size; ++i) {
    unsigned char c;
    Serialize(s, c);
    history.push_back(c);
  }
  for (auto& mem : match_memory) {
    unsigned int size;
    Serialize(s, size);
    if (size < (5.0/9.0) * mem.table.size()) {
      // If the table is sparse, encode keys+values.
      for (int i = 0; i < size; ++i) {
        unsigned int key;
        Serialize(s, key);
        SerializeArray(s, mem.table[key]);
      }
    } else {
      // If the table is dense, encode all values.
      for (int i = 0; i < mem.table.size(); ++i) {
        SerializeArray(s, mem.table[i]);
      }
    }
    SerializeArray(s, mem.predictions);
    SerializeArray(s, mem.counts);
  }
}

void LongTermMemory::Copy(const MemoryInterface* m) {
  const LongTermMemory* orig = static_cast<const LongTermMemory*>(m);
  for (int i = 0; i < indirect.size(); ++i) {
    indirect[i].nonstationary_table = orig->indirect[i].nonstationary_table;
    indirect[i].run_map_table = orig->indirect[i].run_map_table;
    indirect[i].nonstationary_predictions =
        orig->indirect[i].nonstationary_predictions;
    indirect[i].run_map_predictions = orig->indirect[i].run_map_predictions;
  }

  for (int i = 0; i < mixers.size(); ++i) {
    mixers[i].mixer_map.clear();
    for (const auto& it : orig->mixers[i].mixer_map) {
      mixers[i].mixer_map[it.first] =
          std::unique_ptr<MixerData>(new MixerData(it.second->weights.size()));
      mixers[i].mixer_map[it.first]->steps = it.second->steps;
      mixers[i].mixer_map[it.first]->weights = it.second->weights;
    }
  }

  lstm_output_layer = orig->lstm_output_layer;
  for (int i = 0; i < neuron_layer_weights.size(); ++i) {
    neuron_layer_weights[i].weights = orig->neuron_layer_weights[i].weights;
  }
  history = orig->history;
  for (int i = 0; i < match_memory.size(); ++i) {
    match_memory[i].table = orig->match_memory[i].table;
    match_memory[i].predictions = orig->match_memory[i].predictions;
    match_memory[i].counts = orig->match_memory[i].counts;
  }
}