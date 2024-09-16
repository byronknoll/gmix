#ifndef MEMORY_INTERFACE_H_
#define MEMORY_INTERFACE_H_

// This is used for saving/loading model memory. This enables compression
// checkpoints (stopping/resuming) and sharing trained models.
class MemoryInterface {
 public:
  MemoryInterface() {}
  virtual ~MemoryInterface() {}
  virtual void WriteToDisk() = 0;
  virtual void ReadFromDisk() = 0;
};

#endif  // MEMORY_INTERFACE_H_
