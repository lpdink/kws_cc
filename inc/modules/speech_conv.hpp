#ifndef _SPEECH_CONV_HPP_
#define _SPEECH_CONV_HPP_
#include "Eigen/Dense"
#include "ops/conv2d.hpp"
#include "utils.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;
using SpeechBackend::Ops::Conv2D;
namespace SpeechBackend {
namespace Module {

template <typename T>
struct SpeechConvData {
  Conv2D<T> *ln;
  Conv2D<T> *gate;
  SpeechConvData(const int in_channels, const int out_channels,
                 const TwoDim kernel_size, const TwoDim stride,
                 const TwoDim padding, T *model_data, int &offset) {
    ln = new Conv2D<T>(in_channels, out_channels, kernel_size, stride, padding,
                       model_data, offset);
    gate = new Conv2D<T>(in_channels, out_channels, kernel_size, stride,
                         padding, model_data, offset);
  }
  ~SpeechConvData() {
    delete this->ln;
    delete this->gate;
    this->ln = nullptr;
    this->gate = nullptr;
  }
};
template <typename T>
class SpeechConv {
 public:
  SpeechConv(int in_channels, int out_channels, const TwoDim kernel_size,
             const TwoDim stride, const TwoDim padding, T *model_data,
             int &offset);
  ~SpeechConv();
  Matrix<T, Dynamic, Dynamic> forward(
      const Matrix<T, Dynamic, Dynamic> &in_feat);

 private:
  SpeechConvData<T> *self_data;
};

template <typename T>
SpeechConv<T>::SpeechConv(int in_channels, int out_channels,
                          const TwoDim kernel_size, const TwoDim stride,
                          const TwoDim padding, T *model_data, int &offset) {
  this->self_data =
      new SpeechConvData<T>(in_channels, out_channels, kernel_size, stride,
                            padding, model_data, offset);
}

template <typename T>
SpeechConv<T>::~SpeechConv() {
  delete this->self_data;
  this->self_data = nullptr;
}

template <typename T>
Matrix<T, Dynamic, Dynamic> SpeechConv<T>::forward(
    const Matrix<T, Dynamic, Dynamic> &in_feat) {}
}  // namespace Module
}  // namespace SpeechBackend
#endif