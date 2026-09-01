#ifndef MODELS_STATIONARY_MAP_H_
#define MODELS_STATIONARY_MAP_H_

#include <cstdint>
#include <string>
#include <vector>

#include "../model.h"

struct StationaryMapMemory : public MemoryInterface {
  StationaryMapMemory(unsigned int num_contexts, std::string desc);
  std::string description;
  std::vector<uint16_t> table;
  unsigned int num_contexts = 0;

  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;
};

class StationaryMap : public Model {
 public:
  StationaryMap(ShortTermMemory& short_term_memory,
                LongTermMemory& long_term_memory,
                const unsigned int& context,
                unsigned int num_contexts,
                int rate,
                std::string description,
                bool enable_analysis = false);

  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory) override;
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory) override;
  void WriteToDisk(std::ofstream* s) override;
  void ReadFromDisk(std::ifstream* s) override;
  void Copy(const MemoryInterface* m) override;
  unsigned long long GetMemoryUsage(
      const ShortTermMemory& short_term_memory,
      const LongTermMemory& long_term_memory) override;

 private:
  StationaryMapMemory* GetMemory(LongTermMemory& long_term_memory);
  const StationaryMapMemory* GetMemory(
      const LongTermMemory& long_term_memory) const;

  const unsigned int& context_;
  unsigned int num_contexts_;
  int rate_;
  int prediction_index_ = 0;
  int memory_index_ = 0;
  unsigned int last_index_ = 0;
};

#endif  // MODELS_STATIONARY_MAP_H_
