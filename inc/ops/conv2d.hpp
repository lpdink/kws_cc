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
  // input:[batch_size, height, width, channel]
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
static Tensor<T, 2, 1> im2col(const Tensor<T, 4, 1> &input,
                              TwoDim &&kernel_size, TwoDim &&stride,
                              TwoDim &&padding) {
  const int batch_size = input.dimension(0);
  const int channels = input.dimension(1);
  const int height = input.dimension(2);
  const int width = input.dimension(3);
  const int conv2d_rst_height =
      (height + padding.first * 2 - kernel_size.first) / stride.first + 1;
  const int conv2d_rst_width =
      (width + padding.second * 2 - kernel_size.second) / stride.second + 1;
  // 结果矩阵：
  // 行：卷积核扫过的次数，完成x次点乘，等于输出的长宽之积
  // 列：一个卷积核一次点乘所需的所有元素，包含通道
  const int rst_row = (conv2d_rst_height * conv2d_rst_width);
  const int win_nums = kernel_size.first * kernel_size.second;
  const int rst_col = channels * win_nums;

  int w_x, w_y, r_idx, c_idx, channel_idx, filled_channel_num, x, y;
  Tensor<T, 2, 1> rst(rst_col, rst_row);
  for (r_idx = 0; r_idx < rst_row; ++r_idx) {
    // 计算窗口位置
    w_x = r_idx / conv2d_rst_width * stride.first;
    w_y = r_idx % conv2d_rst_width * stride.second;
    for (c_idx = 0; c_idx < rst_col; ++c_idx) {
      channel_idx = c_idx / (win_nums);
      // 计算已经填充了的之前的channel的数量
      filled_channel_num = channel_idx * win_nums;
      x = w_x + (c_idx - filled_channel_num) / kernel_size.second -
          padding.first;
      y = w_y + (c_idx - filled_channel_num) % kernel_size.second -
          padding.second;
      if (x < 0 || y < 0 || x >= height || y >= width) {
        rst(c_idx, r_idx) = 0;
      } else {
        rst(c_idx, r_idx) = input(0, channel_idx, x, y);
      }
    }
  }
  return rst;
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
  Tensor<T, 2, 1> input_col = im2col(input, std::move(kernel_size_),
                                     std::move(stride_), std::move(padding_));
  // Reshape the weight tensor into a matrix
  Tensor<T, 2, 1> weight_col = self_data->w_->reshape(Eigen::array<int, 2>(
      {this->output_channels_,
       input_channels_ * kernel_size_.first * kernel_size_.second}));
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
              j * output_height * output_width + k * output_width + l)+*(this->self_data->b_->data()+j);
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