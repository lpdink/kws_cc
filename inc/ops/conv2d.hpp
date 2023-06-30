#ifndef _CONV2D_HPP_
#define _CONV2D_HPP_
#include "Eigen/Dense"
#include "utils.hpp"

using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;

namespace KwsBackend {

template <typename T>
struct Conv2dData {
  Conv2dData(int input_channels, int output_channels, TwoDim kernel_size,
             T *model_data, int &offset) {
    this->w_ = new Map<Matrix<T, Dynamic, Dynamic>>(
        model_data + offset, input_channels,
        output_channels * kernel_size.first * kernel_size.second);
    offset += output_channels * input_channels * kernel_size.first *
              kernel_size.second;

    this->b_ =
        new Map<Matrix<T, Dynamic, 1>>(model_data + offset, output_channels);
    offset += output_channels;
  }
  ~Conv2dData() {
    delete this->w_;
    delete this->b_;
    this->w_ = nullptr;
    this->b_ = nullptr;
  }

 private:
  Map<Matrix<T, Dynamic, Dynamic>> *w_;  // 卷积核权重矩阵
  Map<Matrix<T, Dynamic, 1>> *b_;        // 偏置向量
};

template <typename T>
class Conv2D {
 public:
  Conv2D(const KwsBackend::Conv2D<float> &) = delete;
  Conv2D(const int input_channels, const int output_channels,
         const TwoDim kernel_size, const TwoDim stride, const TwoDim padding,
         T *model_data, int &offset);

  ~Conv2D();

  Matrix<T, Dynamic, Dynamic> forward(const Matrix<T, Dynamic, Dynamic> &input);

 private:
  int input_channels_;   // 输入通道数
  int output_channels_;  // 输出通道数
  TwoDim kernel_size_;   // 卷积核大小
  TwoDim padding_;       // padding大小
  TwoDim stride_;        // 步长

  Conv2dData<T> *self_data;
};

template <typename T>
Conv2D<T>::Conv2D(const int input_channels, const int output_channels,
                  const TwoDim kernel_size, const TwoDim padding,
                  const TwoDim stride, T *model_data, int &offset)
    : input_channels_(input_channels),
      output_channels_(output_channels),
      kernel_size_(kernel_size),
      padding_(padding),
      stride_(stride) {
  this->self_data = new Conv2dData<T>(input_channels, output_channels,
                                      kernel_size, model_data, offset);
}

template <typename T>
Matrix<T, Dynamic, Dynamic> Conv2D<T>::forward(
    const Matrix<T, Dynamic, Dynamic> &input) {}

template <typename T>
Conv2D<T>::~Conv2D() {
  delete this->self_data;
  this->self_data = nullptr;
}
}  // namespace KwsBackend
#endif