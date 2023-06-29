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
        model_data + offset, output_channels,
        input_channels * kernel_size.first * kernel_size.second);
    offset += output_channels * input_channels * kernel_size.first * kernel_size.second;

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
  Conv2D(int input_channels, int output_channels, TwoDim kernel_size,
         TwoDim padding, TwoDim stride, T *model_data, int &offset);
  //          {
  //     // 初始化卷积核权重矩阵和偏置向量
  //     W_ = Matrix<T, Dynamic, Dynamic>::Random(
  //         output_channels, input_channels * kernel_size * kernel_size);
  //     b_ = Matrix<T, Dynamic, 1>::Random(output_channels);
  //   }

  Matrix<T, Dynamic, Dynamic> forward(const Matrix<T, Dynamic, Dynamic> &input);

 private:
  int input_channels_;   // 输入通道数
  int output_channels_;  // 输出通道数
  TwoDim kernel_size_;      // 卷积核大小
  TwoDim padding_;          // padding大小
  TwoDim stride_;           // 步长

  Conv2dData<T> *self_data;

  //   Matrix<T, Dynamic, Dynamic> W_;  // 卷积核权重矩阵
  //   Matrix<T, Dynamic, 1> b_;               // 偏置向量
};

template <typename T>
Conv2D<T>::Conv2D(int input_channels, int output_channels, TwoDim kernel_size,
         TwoDim padding, TwoDim stride, T *model_data, int &offset)
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
    const Matrix<T, Dynamic, Dynamic> &input) {
  int input_height = input.rows();
    int input_width = input.cols();
    int output_height = (input_height + 2 * padding_.second - kernel_size_.second) / stride_.second + 1;
    int output_width = (input_width + 2 * padding_.first - kernel_size_.first) / stride_.first + 1;
    
    Matrix<T, Dynamic, Dynamic> output(output_height * output_width, output_channels_);

    // Perform convolution
    for (int i = 0; i < output_height; i++) {
        for (int j = 0; j < output_width; j++) {
            for (int c = 0; c < output_channels_; c++) {
                Matrix<T, Dynamic, Dynamic> input_patch = input.block(0, c, kernel_size_.second, kernel_size_.first);
                output(c, i, j) = (input_patch.array() * this->self_data->w_.array().col(c)).sum() + this->self_data->b_(c);
            }
        }
    }
    
    return output;
}
/*
 {
  int input_height = input.rows();
  int input_width = input.cols();

  int output_height =
      (input_height - kernel_size_ + 2 * padding_) / stride_ + 1;
  int output_width = (input_width - kernel_size_ + 2 * padding_) / stride_ + 1;

  Matrix<T, Dynamic, Dynamic> output(output_height, output_width);

  for (int i = 0; i < output_height; i++) {
    for (int j = 0; j < output_width; j++) {
      int start_row = i * stride_ - padding_;
      int start_col = j * stride_ - padding_;
      int end_row = start_row + kernel_size_;
      int end_col = start_col + kernel_size_;

      Matrix<T, Dynamic, Dynamic> input_patch =
          input.block(start_row, start_col, kernel_size_, kernel_size_);
      Matrix<T, Dynamic, 1> input_patch_flattened =
          input_patch.reshape(input_channels_ * kernel_size_ * kernel_size_, 1);
      Matrix<T, Dynamic, 1> result =
          (this->self_data->w_ * input_patch_flattened +
this->self_data->b_).unaryExpr([](T val) { return std::max(val,
static_cast<T>(0));
          });
      output(i, j) = result.sum();
    }
  }

  return output;
}*/
}  // namespace KwsBackend
#endif