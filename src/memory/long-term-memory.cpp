#include "long-term-memory.h"

#include <set>

void LongTermMemory::WriteToDisk(std::ofstream* s) {
  for (auto& mem_ptr : direct) {
    for (auto& pred : mem_ptr->predictions) {
      SerializeArray(s, pred);
    }
    for (auto& counts : mem_ptr->counts) {
      SerializeArray(s, counts);
    }
  }

  for (auto& ptr : mixers) {
    unsigned int mixer_size = ptr->mixer_map.size();
    Serialize(s, mixer_size);
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
}

void LongTermMemory::ReadFromDisk(std::ifstream* s) {
  for (auto& mem_ptr : direct) {
    for (auto& pred : mem_ptr->predictions) {
      SerializeArray(s, pred);
    }
    for (auto& counts : mem_ptr->counts) {
      SerializeArray(s, counts);
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
}