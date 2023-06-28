#ifndef _SPEECH_CONV_H_
#define _SPEECH_CONV_H_
#include "Eigen/Dense"
#include "utils.h"

using Eigen::Dynamic;
using Eigen::Matrix;

class SpeechConv {
 public:
  template <typename T>
  SpeechConv(int in_channels, int out_channels, const TwoDim &kernel_size,
             const TwoDim &stride, const TwoDim &padding, bool is_bn,
             const T &model_data, int &offset);
  ~SpeechConv();
  template <typename T>
  Matrix<T, Dynamic, Dynamic> forward(
      const Matrix<T, Dynamic, Dynamic> &in_feat);

 private:
  void *self_data;
};
#endif