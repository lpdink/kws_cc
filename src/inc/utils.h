#ifndef _KWS_UTILS_H_
#define _KWS_UTILS_H_

float *load_file(const char *file_path);

typedef struct TwoDim {
  TwoDim(int one, int two) : first(one), second(two) {}
  int first;
  int second;
};

#endif