#include "long-term-memory.h"

#include <set>

void LongTermMemory::WriteToDisk(std::ofstream* os) {
  for (auto& mem_ptr : direct) {
    for (auto& pred : mem_ptr->predictions) {
      for (float& p : pred) {
        os->write(reinterpret_cast<char*>(&p), sizeof(p));
      }
    }
    for (auto& counts : mem_ptr->counts) {
      for (unsigned char& c : counts) {
        os->write(reinterpret_cast<char*>(&c), sizeof(c));
      }
    }
  }

  for (auto& ptr : mixers) {
    unsigned int mixer_size = ptr->mixer_map.size();
    os->write(reinterpret_cast<char*>(&mixer_size), sizeof(mixer_size));
    unsigned int input_size = ptr->mixer_map.begin()->second->weights.size();
    os->write(reinterpret_cast<char*>(&(input_size)), sizeof(input_size));
    std::set<unsigned int> keys;  // use a set to get consistent key order.
    for (auto& it : ptr->mixer_map) {
      keys.insert(it.first);
    }
    for (unsigned int context : keys) {
      os->write(reinterpret_cast<char*>(&context), sizeof(context));
      os->write(reinterpret_cast<char*>(&(ptr->mixer_map[context]->steps)),
                sizeof(ptr->mixer_map[context]->steps));
      for (float p : ptr->mixer_map[context]->weights) {
        os->write(reinterpret_cast<char*>(&p), sizeof(p));
      }
    }
  }

  for (auto& x : lstm_output_layer) {
    for (auto& y : x) {
      for (float& z : y) {
        os->write(reinterpret_cast<char*>(&z), sizeof(z));
      }
    }
  }
  for (auto& x : neuron_layer_weights) {
    for (auto& y : x->weights) {
      for (float& z : y) {
        os->write(reinterpret_cast<char*>(&z), sizeof(z));
      }
    }
  }
}

void LongTermMemory::ReadFromDisk(std::ifstream* is) {
  for (auto& mem_ptr : direct) {
    for (auto& pred : mem_ptr->predictions) {
      for (float& p : pred) {
        is->read(reinterpret_cast<char*>(&p), sizeof(p));
      }
    }
    for (auto& counts : mem_ptr->counts) {
      for (unsigned char& c : counts) {
        is->read(reinterpret_cast<char*>(&c), sizeof(c));
      }
    }
  }

  for (auto& ptr : mixers) {
    ptr->mixer_map.clear();
    unsigned int mixer_size;
    is->read(reinterpret_cast<char*>(&mixer_size), sizeof(mixer_size));
    unsigned int input_size;
    is->read(reinterpret_cast<char*>(&(input_size)), sizeof(input_size));
    for (int i = 0; i < mixer_size; ++i) {
      unsigned int context;
      is->read(reinterpret_cast<char*>(&context), sizeof(context));
      ptr->mixer_map[context] =
          std::unique_ptr<MixerData>(new MixerData(input_size));
      is->read(reinterpret_cast<char*>(&(ptr->mixer_map[context]->steps)),
               sizeof(ptr->mixer_map[context]->steps));
      for (int j = 0; j < input_size; ++j) {
        float p;
        is->read(reinterpret_cast<char*>(&p), sizeof(p));
        ptr->mixer_map[context]->weights[j] = p;
      }
    }
  }

  for (auto& x : lstm_output_layer) {
    for (auto& y : x) {
      for (float& z : y) {
        is->read(reinterpret_cast<char*>(&z), sizeof(z));
      }
    }
  }
  for (auto& x : neuron_layer_weights) {
    for (auto& y : x->weights) {
      for (float& z : y) {
        is->read(reinterpret_cast<char*>(&z), sizeof(z));
      }
    }
  }
}