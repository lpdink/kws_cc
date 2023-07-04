#ifndef _CONV2D_HPP_
#define _CONV2D_HPP_
#include <iostream>

#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils.hpp"

using Eigen::Dynamic;
using Eigen::Tensor;
using Eigen::TensorMap;

namespace SpeechBackend {
namespace Ops {

template <typename T>
struct Conv2dData {
  Conv2dData(int input_channels, int output_channels, TwoDim kernel_size,
             T *model_data, int &offset) {
    this->w_ = new TensorMap<Tensor<T, 4, 1>>(
        model_data + offset, output_channels, input_channels, kernel_size.first,
        kernel_size.second);
    offset += output_channels * input_channels * kernel_size.first *
              kernel_size.second;

    this->b_ =
        new TensorMap<Tensor<T, 1, 1>>(model_data + offset, output_channels);
    offset += output_channels;
  }
  ~Conv2dData() {
    delete this->w_;
    delete this->b_;
    this->w_ = nullptr;
    this->b_ = nullptr;
  }
  TensorMap<Tensor<T, 4, 1>> *w_;  // weight: [out_channel, in_channel,
                                   // kernel_size.first, kernel_size.second]
  TensorMap<Tensor<T, 1, 1>> *b_;  // bias:[out_channel]
};

template <typename T>
class Conv2D {
 public:
  Conv2D(const Conv2D<float> &) = delete;
  Conv2D(const int input_channels, const int output_channels,
         const TwoDim &&kernel_size, const TwoDim &&stride,
         const TwoDim &&padding, T *model_data, int &offset);

  ~Conv2D();
  // input:[batch_size, channel, height, width]
  Tensor<T, 4, 1> forward(const Tensor<T, 4, 1> &input);

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
                  const TwoDim &&kernel_size, const TwoDim &&stride,
                  const TwoDim &&padding, T *model_data, int &offset)
    : input_channels_(input_channels),
      output_channels_(output_channels),
      kernel_size_(kernel_size),
      padding_(padding),
      stride_(stride) {
  this->self_data = new Conv2dData<T>(input_channels, output_channels,
                                      kernel_size, model_data, offset);
}

template <typename T>
static Tensor<T, 2, 1> im2col(const Tensor<T, 4, 1> &input, int kernel_height,
                              int kernel_width, int padding_height,
                              int padding_width, int stride_height,
                              int stride_width) {
  int batch_size = input.dimension(0);
  int input_channels = input.dimension(1);
  int input_height = input.dimension(2);
  int input_width = input.dimension(3);
  int output_height =
      (input_height - kernel_height + 2 * padding_height) / stride_height + 1;
  int output_width =
      (input_width - kernel_width + 2 * padding_width) / stride_width + 1;
  int col_height = kernel_height * kernel_width * input_channels;
  int col_width = output_height * output_width;

  Tensor<T, 2, 1> col(col_height, col_width);

  for (int c = 0; c < col_height; ++c) {
    int channel = c % input_channels;
    int kernel_row = (c / input_channels) / kernel_width;
    int kernel_col = (c / input_channels) % kernel_width;

    for (int h = 0; h < output_height; ++h) {
      for (int w = 0; w < output_width; ++w) {
        int input_row = h * stride_height - padding_height + kernel_row;
        int input_col = w * stride_width - padding_width + kernel_col;

        if (input_row >= 0 && input_row < input_height && input_col >= 0 &&
            input_col < input_width) {
          col(c, h * output_width + w) =
              input(0, channel, input_row, input_col);
        } else {
          col(c, h * output_width + w) = 0;
        }
      }
    }
  }

  return col;
}

template <typename T>
Tensor<T, 4, 1> Conv2D<T>::forward(const Tensor<T, 4, 1> &input) {
  int batch_size = input.dimension(0);
  int input_height = input.dimension(2);
  int input_width = input.dimension(3);
  int output_height =
      (input_height - kernel_size_.first + 2 * padding_.first) / stride_.first +
      1;
  int output_width = (input_width - kernel_size_.second + 2 * padding_.second) /
                         stride_.second +
                     1;

  // Reshape the input tensor using im2col
  // 60, 8192
  Tensor<T, 2, 1> input_col =
      im2col(input, kernel_size_.first, kernel_size_.second, padding_.first,
             padding_.second, stride_.first, stride_.second);

  // Reshape the weight tensor into a matrix
  Tensor<T, 2, 1> weight_col = self_data->w_->reshape(Eigen::array<T, 2>(
      {this->output_channels_,
       input_channels_ * kernel_size_.first * kernel_size_.second}));
  std::cout << weight_col << std::endl;
  std::cout << "---------" << std::endl;
  std::cout << input_col << std::endl;
  // Perform the matrix multiplication
  Tensor<T, 2, 1> output_col = weight_col.contract(
      input_col,
      Eigen::array<Eigen::IndexPair<int>, 1>{{Eigen::IndexPair<int>(1, 0)}});

  // Reshape the output matrix back into a tensor
  Tensor<T, 4, 1> output(batch_size, output_channels_, output_height,
                         output_width);
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < output_channels_; ++j) {
      for (int k = 0; k < output_height; ++k) {
        for (int l = 0; l < output_width; ++l) {
          output(i, j, k, l) = output_col(
              i * output_channels_ * output_height * output_width +
              j * output_height * output_width + k * output_width + l);
        }
      }
    }
  }

  return output;
}

template <typename T>
Conv2D<T>::~Conv2D() {
  delete this->self_data;
  this->self_data = nullptr;
}
}  // namespace Ops
}  // namespace SpeechBackend
#endif