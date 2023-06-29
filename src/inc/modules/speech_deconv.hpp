#ifndef _SPEECH_DECONV_HPP_
#define _SPEECH_DECONV_HPP_
#include "Eigen/Dense"
#include "utils.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;
namespace KwsBackend {

template <typename T>
class SpeechDeConv {
 public:
  SpeechDeConv(int in_channels, int out_channels, const TwoDim &kernel_size,
               const TwoDim &stride, const TwoDim &padding, bool is_bn,
               const T &model_data, int &offset);
  ~SpeechDeConv();
  Matrix<T, Dynamic, Dynamic> forward(
      const Matrix<T, Dynamic, Dynamic> &in_feat);

 private:
  void *self_data;
};
template <typename T>
SpeechDeConv<T>::SpeechDeConv(int in_channels, int out_channels,
                              const TwoDim &kernel_size, const TwoDim &stride,
                              const TwoDim &padding, bool is_bn,
                              const T &model_data, int &offset) {}

template <typename T>
SpeechDeConv<T>::~SpeechDeConv() {}

template <typename T>
Matrix<T, Dynamic, Dynamic> SpeechDeConv<T>::forward(
    const Matrix<T, Dynamic, Dynamic> &in_feat) {}
}  // namespace KwsBackend
#endif