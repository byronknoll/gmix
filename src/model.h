#ifndef MODEL_H_
#define MODEL_H_

#include "long-term-memory.h"
#include "short-term-memory.h"

// Models are used to make predictions. There are two types of data models can
// store:
// 1) Long-term memory. See the LongTermMemory struct (long-term-memory.h).
// 2) Short-term memory. Models can either use class member variables or the
// ShortTermMemory struct (see short-term-memory.h).
//
// This separation of data is important, so that models can train/learn their
// long-term memory, and separately use their short-term memory for
// prediction/inference (with long-term memory frozen).
//
// Models always "Predict", "Perceive", and then optionally "Learn". See below
// for descriptions of these three functions.
class Model {
 public:
  Model() {}
  virtual ~Model() {}
  // "Predict" is called before the next bit of the input sequence is seen.
  // Models can make a prediction about the next bit in this function.
  virtual void Predict(ShortTermMemory& short_term_memory,
                       const LongTermMemory& long_term_memory) = 0;
  // "Perceive" is called after the next bit of the input sequence is seen.
  // Models can update their state (short-term memory) based on this bit.
  virtual void Perceive(ShortTermMemory& short_term_memory,
                        const LongTermMemory& long_term_memory) = 0;
  // "Learn" can be called after "Perceive", allowing models to update their
  // long-term memory. Note: "Learn" is not always called. Models should be able
  // to make meaningful predictions even when long-term memory is frozen.
  virtual void Learn(const ShortTermMemory& short_term_memory,
                     LongTermMemory& long_term_memory) = 0;
};

#endif  // MODEL_H_
