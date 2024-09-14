#ifndef MEMORY_INTERFACE_H_
#define MEMORY_INTERFACE_H_

class MemoryInterface {
 public:
  MemoryInterface() {}
  virtual ~MemoryInterface() {}
  virtual void WriteToDisk() = 0;
  virtual void ReadFromDisk() = 0;
};

#endif // MEMORY_INTERFACE_H_
