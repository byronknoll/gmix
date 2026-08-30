#include "long-term-memory.h"

void LongTermMemory::WriteToDisk(std::ofstream* s) {
  unsigned long long size = history.size();
  Serialize(s, size);
  for (unsigned long long i = 0; i < size; ++i) {
    Serialize(s, history[i]);
  }
  for (auto& m : model_memory) {
    m->WriteToDisk(s);
  }
}

void LongTermMemory::ReadFromDisk(std::ifstream* s) {
  unsigned long long size;
  history.clear();
  Serialize(s, size);
  for (unsigned long long i = 0; i < size; ++i) {
    unsigned char c;
    Serialize(s, c);
    history.push_back(c);
  }
  for (auto& m : model_memory) {
    m->ReadFromDisk(s);
  }
}

void LongTermMemory::Copy(const MemoryInterface* m) {
  const LongTermMemory* orig = static_cast<const LongTermMemory*>(m);
  history = orig->history;
  for (size_t i = 0; i < model_memory.size(); ++i) {
    model_memory[i]->Copy(orig->model_memory[i].get());
  }
}