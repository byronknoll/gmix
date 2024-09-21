#ifndef DECODER_H
#define DECODER_H

#include <fstream>

#include "../predictor.h"

class Decoder {
 public:
  // Set "resume" to true if restarting from a checkpoint.
  Decoder(std::ifstream* is, Predictor* p, bool resume = false);
  int Decode();
  void WriteCheckpoint(std::string path);
  void ReadCheckpoint(std::string path);

 private:
  int ReadByte();
  unsigned int Discretize(float p);

  std::ifstream* is_;
  unsigned int x1_, x2_, x_;
  Predictor* p_;
};

#endif
