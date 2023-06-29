#ifndef _SPEECH_CONV_HPP_
#define _SPEECH_CONV_HPP_
#include "Eigen/Dense"
#include "utils.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;
namespace KwsBackend {
template <typename T>
class SpeechConv {
 public:
  SpeechConv(int in_channels, int out_channels, const TwoDim &kernel_size,
             const TwoDim &stride, const TwoDim &padding, bool is_bn,
             const T *model_data, int &offset);
  ~SpeechConv();
  Matrix<T, Dynamic, Dynamic> forward(
      const Matrix<T, Dynamic, Dynamic> &in_feat);

 private:
  void *self_data;
};

template <typename T>
SpeechConv<T>::SpeechConv(int in_channels, int out_channels,
                          const TwoDim &kernel_size, const TwoDim &stride,
                          const TwoDim &padding, bool is_bn,
                          const T *model_data, int &offset) {}

template <typename T>
SpeechConv<T>::~SpeechConv() {}

template <typename T>
Matrix<T, Dynamic, Dynamic> SpeechConv<T>::forward(
    const Matrix<T, Dynamic, Dynamic> &in_feat) {}
}  // namespace KwsBackend
#endif