#ifndef _CRN_UTILS_HPP_
#define _CRN_UTILS_HPP_
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace SpeechBackend {
template <typename T>
T *load_file(const char *file_path, int &size) {
  int ret = -1;
  size = 0;
  struct stat st;
  T *rst = nullptr;
  ret = stat(file_path, &st);
  if (ret == 1) {
    printf("file %s open failed.", file_path);
    exit(0);
  } else {
    size = st.st_size;
    FILE *file = fopen(file_path, "rb");
    rst = new T[size];
    size = fread(rst, sizeof(T), size, file);
    fclose(file);
  }
  return rst;
}

template <typename T>
void write_out(const char *file_name, T *ptr, int size) {
  FILE *file = fopen(file_name, "wb");  // Open the file in binary mode
  if (!file) {
    printf("write_out failed. file %s open failed.", file_name);
    // handle file opening error
    return;
  }

  // write the array to the file
  fwrite(ptr, sizeof(T), size, file);

  // close the file
  fclose(file);
}

struct TwoDim {
  TwoDim(int one, int two) : first(one), second(two) {}
  int first;
  int second;
};
}  // namespace SpeechBackend
#endif