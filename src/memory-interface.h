#ifndef MEMORY_INTERFACE_H_
#define MEMORY_INTERFACE_H_

#include <fstream>

// This is used for saving/loading model memory. This enables compression
// checkpoints (stopping/resuming) and sharing trained models.
class MemoryInterface {
 public:
  MemoryInterface() {}
  virtual ~MemoryInterface() {}
  virtual void WriteToDisk(std::ofstream* s) = 0;
  virtual void ReadFromDisk(std::ifstream* s) = 0;
  // This creates a deep copy of another memory. This should be equivalent to
  // "WriteToDisk" (for one memory) followed by "ReadFromDisk" (for the other
  // memory), except the copying takes place in RAM instead of disk (which is
  // faster, but needs twice as much memory).
  virtual void Copy(const MemoryInterface* m) = 0;

 protected:
  template <typename T>
  void SerializeArray(std::ofstream* os, T& x) {
    os->write(reinterpret_cast<char*>(&x[0]), x.size() * sizeof(x[0]));
  }
  template <typename T>
  void SerializeArray(std::ifstream* is, T& x) {
    is->read(reinterpret_cast<char*>(&x[0]), x.size() * sizeof(x[0]));
  }
  template <typename T>
  void Serialize(std::ofstream* os, T& x) {
    os->write(reinterpret_cast<char*>(&x), sizeof(x));
  }
  template <typename T>
  void Serialize(std::ifstream* is, T& x) {
    is->read(reinterpret_cast<char*>(&x), sizeof(x));
  }
};

#endif  // MEMORY_INTERFACE_H_
