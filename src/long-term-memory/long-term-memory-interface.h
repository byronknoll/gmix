#ifndef LONG_TERM_MEMORY_INTERFACE_H_
#define LONG_TERM_MEMORY_INTERFACE_H_

class LongTermMemoryInterface {
 public:
  LongTermMemoryInterface() {}
  virtual ~LongTermMemoryInterface() {}
  virtual void WriteToDisk() = 0;
  virtual void ReadFromDisk() = 0;
};

#endif // LONG_TERM_MEMORY_INTERFACE_H_
