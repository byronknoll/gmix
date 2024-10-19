#ifndef SIGMOID_H
#define SIGMOID_H

#include <vector>

class Sigmoid {
 public:
  Sigmoid(int logit_size);
  float Logit(float p) const;
  static float Logistic(float p);
  static float SlowLogit(float p);

 private:
  int logit_size_;
  std::vector<float> logit_table_;
};

#endif
