#include "sigmoid.h"

#include <math.h>

float Sigmoid::Logistic(float p) { return 1 / (1 + exp(-p)); }

float Sigmoid::Logit(float p) {
  if (p < 0.0001)
    p = 0.0001;
  else if (p > 0.9999)
    p = 0.9999;
  return log(p / (1 - p));
}
