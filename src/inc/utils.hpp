#ifndef _KWS_UTILS_HPP_
#define _KWS_UTILS_HPP_

namespace KwsBackend {
float *load_file(const char *file_path);

typedef struct TwoDim {
  TwoDim(int one, int two) : first(one), second(two) {}
  int first;
  int second;
};
}  // namespace KwsBackend
#endif