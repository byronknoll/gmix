#ifndef SIGMOID_H
#define SIGMOID_H

// Logit and logistic functions.
class Sigmoid {
 public:
  // Input: logit space.
  // Output: probability (0-1).
  static float Logistic(float p);
  // Input: probability (0-1).
  // Output: logit space.
  static float Logit(float p);
};

#endif
