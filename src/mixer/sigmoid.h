#ifndef SIGMOID_H
#define SIGMOID_H

#include <vector>

// Logit and logistic functions.
class Sigmoid {
 public:
  // The number of table entries. More entries = more accurate (but uses more memory).
  Sigmoid(int logit_size);
  // This computes fast logit using a table lookup.
  // Input: probability (0-1).
  // Output: logit space.
  float Logit(float p) const;
  // Input: logit space.
  // Output: probability (0-1).
  static float Logistic(float p);
  static float SlowLogit(float p);

 private:
  int logit_size_;
  std::vector<float> logit_table_;
};

#endif
