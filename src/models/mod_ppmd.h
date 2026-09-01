#ifndef MODELS_MOD_PPMD_H
#define MODELS_MOD_PPMD_H

#include <memory>

#include "../model.h"

namespace PPMD {

class ppmd_Model;

class ModPPMD : public Model {
 public:
  ModPPMD(ShortTermMemory& short_term_memory, LongTermMemory& long_term_memory,
          int order, int memory, bool enable_analysis,
          bool update_shared_ppm_predictions = true);
  void Predict(ShortTermMemory& short_term_memory,
               const LongTermMemory& long_term_memory);
  void Learn(const ShortTermMemory& short_term_memory,
             LongTermMemory& long_term_memory);
  void WriteToDisk(std::ofstream* s);
  void ReadFromDisk(std::ifstream* s);
  void Copy(const MemoryInterface* m);
  unsigned long long GetMemoryUsage(const ShortTermMemory& short_term_memory,
                                    const LongTermMemory& long_term_memory);

 private:
  // The PPM tree/heap is the model's long-term (trainable) memory, so it lives
  // in LongTermMemory (see memory_index_) instead of being owned here. This
  // accessor pair enforces that "Predict" can only reach the const, read-only
  // surface of the model, while "Learn" gets mutable access to actually update
  // it.
  ppmd_Model* GetModel(LongTermMemory& long_term_memory);
  const ppmd_Model* GetModel(const LongTermMemory& long_term_memory) const;

  int memory_index_;
  // Cached so WriteToDisk/ReadFromDisk/Copy (which don't receive
  // LongTermMemory per the Model interface) can still reach GetModel() to
  // serialize MaxContext's offset as short-term state. Valid for this
  // object's lifetime: Predictor destroys models_ before long_term_memory_.
  LongTermMemory* long_term_memory_;
  // True once the current byte's context has already been advanced (either
  // by "Learn" doing the real update, or by "Predict" doing a no-learn
  // advance), so the other one doesn't redundantly (and incorrectly) try
  // to advance again for the same byte. Starts true because the constructor
  // already primes the model for the first byte.
  bool context_advanced_ = true;
  bool update_shared_ppm_predictions_ = true;
  std::valarray<float> byte_predictions_;
  // top_, mid_, and bot_ are used to keep track of ranges for converting
  // byte-level predictions to bit-level predictions. The range is updated as
  // bits are observed.
  int top_, mid_, bot_, prediction_index_;
};

}  // namespace PPMD

#endif  // MODELS_MOD_PPMD_H
