#ifndef MODELS_MOD_PPMD_H
#define MODELS_MOD_PPMD_H

#include <memory>

#include "../model.h"

namespace PPMD {

class ppmd_Model;

class ModPPMD : public Model {
 public:
  ModPPMD(ShortTermMemory& short_term_memory,
            LongTermMemory& long_term_memory, int order, int memory);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory) {}
  void WriteToDisk(std::ofstream* s);
  void ReadFromDisk(std::ifstream* s);
  void Copy(const MemoryInterface* m);

 private:
  std::unique_ptr<ppmd_Model> ppmd_model_;
  // top_, mid_, and bot_ are used to keep track of ranges for converting
  // byte-level predictions to bit-level predictions. The range is updated as
  // bits are observed.
  int top_, mid_, bot_, prediction_index_;
};

}  // namespace PPMD

#endif  // MODELS_MOD_PPMD_H
