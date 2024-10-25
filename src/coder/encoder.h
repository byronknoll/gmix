#ifndef ENCODER_H
#define ENCODER_H

#include <fstream>

#include "../predictor.h"

class Encoder {
 public:
  Encoder(std::ofstream* os);
  void Encode(int bit, float prediction);
  void Flush();
  void WriteCheckpoint(std::string path);
  void ReadCheckpoint(std::string path);

 private:
  void WriteByte(unsigned int byte);
  unsigned int Discretize(float p);

  std::ofstream* os_;
  unsigned int x1_, x2_;
};

#endif
