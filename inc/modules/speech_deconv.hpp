#ifndef _SPEECH_DECONV_HPP_
#define _SPEECH_DECONV_HPP_
#include "Eigen/Dense"
#include "ops/conv2d_t.hpp"
#include "utils.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;
using SpeechBackend::Ops::ConvTranspose2D;

namespace SpeechBackend {
namespace Module {

template <typename T>
struct SpeechDeConvData {
  ConvTranspose2D<T> *ln;
  ConvTranspose2D<T> *gate;
  SpeechDeConvData(const int in_channels, const int out_channels,
                   const TwoDim kernel_size, const TwoDim stride,
                   const TwoDim padding, T *model_data, int &offset) {
    ln = new ConvTranspose2D<T>(in_channels, out_channels, kernel_size, stride,
                                padding, model_data, offset);
    gate = new ConvTranspose2D<T>(in_channels, out_channels, kernel_size,
                                  stride, padding, model_data, offset);
  }
  ~SpeechDeConvData() {
    delete this->ln;
    delete this->gate;
    this->ln = nullptr;
    this->gate = nullptr;
  }
};
template <typename T>
class SpeechDeConv {
 public:
  SpeechDeConv(int in_channels, int out_channels, const TwoDim kernel_size,
               const TwoDim stride, const TwoDim padding, T *model_data,
               int &offset);
  ~SpeechDeConv();
  Matrix<T, Dynamic, Dynamic> forward(
      const Matrix<T, Dynamic, Dynamic> &in_feat);

 private:
  SpeechDeConvData<T> *self_data;
};

template <typename T>
SpeechDeConv<T>::SpeechDeConv(int in_channels, int out_channels,
                              const TwoDim kernel_size, const TwoDim stride,
                              const TwoDim padding, T *model_data,
                              int &offset) {
  this->self_data =
      new SpeechDeConvData<T>(in_channels, out_channels, kernel_size, stride,
                              padding, model_data, offset);
}

template <typename T>
SpeechDeConv<T>::~SpeechDeConv() {
  delete this->self_data;
  this->self_data = nullptr;
}

template <typename T>
Matrix<T, Dynamic, Dynamic> SpeechDeConv<T>::forward(
    const Matrix<T, Dynamic, Dynamic> &in_feat) {}
}  // namespace Module
}  // namespace SpeechBackend
#endif