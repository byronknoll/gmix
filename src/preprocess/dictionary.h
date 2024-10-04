#ifndef DICTIONARY_H
#define DICTIONARY_H

#include <stdio.h>

#include <array>
#include <list>
#include <string>
#include <unordered_map>

namespace preprocessor {

// This performs dictionary preprocessing using a word replacing transform.
// You can run this using the "prep" tool (runner/prep.cpp).
class Dictionary {
 public:
  Dictionary(FILE* dictionary, bool encode, bool decode);
  void Encode(FILE* input, int len, FILE* output);
  void Decode(FILE* input, FILE* output);

 private:
  bool DecodeChar(FILE* input, unsigned char& output);
  void EncodeWord(const std::string& word, int num_upper, bool next_lower,
                  FILE* output);
  bool EncodeSubstring(const std::string& word, FILE* output);
  bool AddToBuffer(FILE* input);

  std::unordered_map<std::string, unsigned int> byte_map_;
  std::unordered_map<unsigned int, std::string> reverse_map_;
  std::list<unsigned char> output_buffer_;
  bool decode_upper_ = false, decode_capital_ = false;
  unsigned int longest_word_ = 0;
};

}  // namespace preprocessor

#endif  // DICTIONARY_H
