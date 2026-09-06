#ifndef SIGMOID_H
#define SIGMOID_H

#include <cmath>
#include <cstdint>

#include "fastmath.h"

// Logit and logistic functions.
class Sigmoid {
 public:
  // Input: logit space.
  // Output: probability (0-1).
  static inline float Logistic(float p) {
    return 1.0f / (1.0f + fast_expf(-p));
  }

  // Input: probability (0-1).
  // Output: logit space.
  static inline float Logit(float p) {
    if (p < 0.0001f)
      p = 0.0001f;
    else if (p > 0.9999f)
      p = 0.9999f;
    return std::log(p / (1.0f - p));
  }
};

#endif
