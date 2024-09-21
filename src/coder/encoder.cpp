#include "encoder.h"

Encoder::Encoder(std::ofstream* os, Predictor* p)
    : os_(os), x1_(0), x2_(0xffffffff), p_(p) {}

void Encoder::WriteByte(unsigned int byte) { os_->put(byte); }

unsigned int Encoder::Discretize(float p) { return 1 + 65534 * p; }

void Encoder::Encode(int bit) {
  const unsigned int p = Discretize(p_->Predict());
  const unsigned int xmid =
      x1_ + ((x2_ - x1_) >> 16) * p + (((x2_ - x1_) & 0xffff) * p >> 16);
  if (bit) {
    x2_ = xmid;
  } else {
    x1_ = xmid + 1;
  }
  p_->Perceive(bit);
  p_->Learn();

  while (((x1_ ^ x2_) & 0xff000000) == 0) {
    WriteByte(x2_ >> 24);
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
}

void Encoder::Flush() {
  while (((x1_ ^ x2_) & 0xff000000) == 0) {
    WriteByte(x2_ >> 24);
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
  WriteByte(x2_ >> 24);
}

void Encoder::WriteCheckpoint(std::string path) {
  std::ofstream data_out(path, std::ios::out | std::ios::binary);
  if (!data_out.is_open()) return;
  data_out.write(reinterpret_cast<char*>(&x1_), sizeof(x1_));
  data_out.write(reinterpret_cast<char*>(&x2_), sizeof(x2_));
  data_out.close();
  os_->flush();
}

void Encoder::ReadCheckpoint(std::string path) {
  std::ifstream data_in(path, std::ios::in | std::ios::binary);
  if (!data_in.is_open()) return;
  data_in.read(reinterpret_cast<char*>(&x1_), sizeof(x1_));
  data_in.read(reinterpret_cast<char*>(&x2_), sizeof(x2_));
  data_in.close();
}