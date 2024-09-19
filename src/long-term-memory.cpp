#include "long-term-memory.h"

void LongTermMemory::WriteToDisk(std::ofstream* os) {
  for (auto& ptr : direct) {
    for (auto pred : ptr->predictions) {
      for (float p : pred) {
        os->write(reinterpret_cast<char*>(&p), sizeof(p));
      }
    }
    for (auto pred : ptr->counts) {
      for (unsigned char c : pred) {
        os->write(reinterpret_cast<char*>(&c), sizeof(c));
      }
    }
  }

  for (auto& ptr : mixers) {
    unsigned int mixer_size = ptr->mixer_map.size();
    os->write(reinterpret_cast<char*>(&mixer_size), sizeof(mixer_size));
    unsigned int input_size = ptr->mixer_map.begin()->second->weights.size();
    os->write(reinterpret_cast<char*>(&(input_size)), sizeof(input_size));
    for (auto& it : ptr->mixer_map) {
      unsigned int context = it.first;
      os->write(reinterpret_cast<char*>(&context), sizeof(context));
      os->write(reinterpret_cast<char*>(&(it.second->steps)),
                sizeof(it.second->steps));
      for (float p : it.second->weights) {
        os->write(reinterpret_cast<char*>(&p), sizeof(p));
      }
    }
  }
}
void LongTermMemory::ReadFromDisk(std::ifstream* is) {
  for (auto& ptr : direct) {
    for (auto pred : ptr->predictions) {
      for (float& p : pred) {
        is->read(reinterpret_cast<char*>(&p), sizeof(p));
      }
    }
    for (auto pred : ptr->counts) {
      for (unsigned char& c : pred) {
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
}