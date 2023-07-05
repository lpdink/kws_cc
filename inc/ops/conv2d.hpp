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

// template <typename T>
// Tensor<T, 2, 1> im2col(const Tensor<T, 4, 1> &input, int kernel_height,
//                               int kernel_width, int padding_height,
//                               int padding_width, int stride_height,
//                               int stride_width) {
//   int batch_size = input.dimension(0); // assert batch_size==1
//   int input_channels = input.dimension(1);
//   int input_height = input.dimension(2);
//   int input_width = input.dimension(3);
//   int output_height =
//       (input_height - kernel_height + 2 * padding_height) / stride_height +
//       1;
//   int output_width =
//       (input_width - kernel_width + 2 * padding_width) / stride_width + 1;
//   int col_height = kernel_height * kernel_width * input_channels; // 60
//   int col_width = output_height * output_width;

//   Tensor<T, 2, 1> col(col_height, col_width); // 60*xx

// //   for(int rdx = 0;rdx<col_height;rdx++){
// //     for(int cdx = 0;cdx<col_width;cdx++){
// //         col(rdx, cdx) = input(0, )
// //     }
// //   }

//   for (int c = 0; c < col_height; ++c) { // 60
//     int channel = c % input_channels;
//     int kernel_row = (c / input_channels) / kernel_width;
//     int kernel_col = (c / input_channels) % kernel_width;

//     for (int h = 0; h < output_height; ++h) {
//       for (int w = 0; w < output_width; ++w) {
//         int input_row = h * stride_height - padding_height + kernel_row;
//         int input_col = w * stride_width - padding_width + kernel_col;

//         if (input_row >= 0 && input_row < input_height && input_col >= 0 &&
//             input_col < input_width) {
//           col(c, h * output_width + w) =
//               input(0, channel, input_row, input_col);
//         } else {
//           col(c, h * output_width + w) = 0;
//         }
//       }
//     }
//   }

//   return col;
// }

//  第二版，参考纯C实现
// template <typename T>
// static Tensor<T, 2, 1> im2col(const Tensor<T, 4, 1> &input,
//                               TwoDim &&kernel_size, TwoDim &&stride,
//                               TwoDim &&padding) {
//   int batch_size = input.dimension(0);
//   int in_channel = input.dimension(3);
//   int in_height = input.dimension(1);
//   int in_width = input.dimension(2);

//   // conv_out_height
//   int win_h =
//       (in_height + 2 * padding.first - kernel_size.first) / stride.first + 1;
//   // conv_out_width
//   int win_w =
//       (in_width + 2 * padding.second - kernel_size.second) / stride.second +
//       1;

//   // 一定范围内卷积核的权重压成一个col_h
//   int col_h = kernel_size.first * kernel_size.second * in_channel;
//   int col_w = win_h * win_w;
//   int x, y;

//   // weight shape==[kernel_nums, col_h]
//   Tensor<T, 2, 1> rst(col_h, col_w);
//   rst.setZero();

//   for (int i = 0; i < col_h; i++) {
//     x = i % win_w;
//     y = i / win_w;
//     for (int j = 0; j < col_w; j++) {
//       int c = j / (kernel_size.second * kernel_size.first);
//       int kj = j % kernel_size.second;
//       int ki = j / kernel_size.second;

//       int row = y * stride.first + ki - padding.first;
//       int col = x * stride.second + kj - padding.second;

//       row = row - padding.first;
//       col = col - padding.second;
//       if (row < 0 || row >= in_height || col < 0 || col >= in_width) {
//         rst(i, j) = 0;
//       } else {
//         rst(i, j) =
//             *(input.data() + c * in_width * in_height + row * in_width +
//             col);
//       }
//     }
//   }
//   return rst;
// }

/*第三版 caffe实现*/
// template <typename T>
// Tensor<T, 2, 1> im2col(const Tensor<T, 4, 1>& input, TwoDim&& kernel_size,
//                        TwoDim&& stride, TwoDim&& padding) {
//   int batch_size = input.dimension(0);
//   int channels = input.dimension(1);
//   int height = input.dimension(2);
//   int width = input.dimension(3);
//   const int output_h = (height + 2 * padding.first - kernel_size.first ) /
//   stride.first+1; const int output_w = (width + 2 * padding.second -
//   kernel_size.second ) / stride.second+1; const int channel_size = height *
//   width; auto *data_im = input.data(); int col_h = kernel_size.first *
//   kernel_size.second * channels; int col_w = output_h * output_w; Tensor<T,
//   2, 1> rst(col_h, col_w); rst.setZero(); auto *data_col = rst.data(); for
//   (int channel = channels; channel--; data_im += channel_size) {
//     for (int kernel_row = 0; kernel_row < kernel_size.first; kernel_row++) {
//       for (int kernel_col = 0; kernel_col < kernel_size.second; kernel_col++)
//       {
//         int input_row = -padding.first + kernel_row;
//         for (int output_rows = output_h; output_rows; output_rows--) {
//           if (!(input_row<height)) {
//             for (int output_cols = output_w; output_cols; output_cols--) {
//               *(data_col++) = 0;
//             }
//           } else {
//             int input_col = -padding.second + kernel_col;
//             for (int output_col = output_w; output_col; output_col--) {
//               if (input_col<width) {
//                 *(data_col++) = data_im[input_row * width + input_col];
//               } else {
//                 *(data_col++) = 0;
//               }
//               input_col += stride.second;
//             }
//           }
//           input_row += stride.first;
//         }
//       }
//     }
//   }
//   return rst;
// }

// 第四版：自己实现
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
  // 60, 8192
  Tensor<T, 2, 1> input_col = im2col(input, std::move(kernel_size_),
                                     std::move(stride_), std::move(padding_));
  // Reshape the weight tensor into a matrix
  Tensor<T, 2, 1> weight_col = self_data->w_->reshape(Eigen::array<T, 2>(
      {this->output_channels_,
       input_channels_ * kernel_size_.first * kernel_size_.second}));
  //   std::cout << "input_cc:" << std::endl << input << std::endl;
  //   std::cout << "weight_col:" << std::endl << weight_col << std::endl;
  //   std::cout << "---------" << std::endl;
  //   std::cout << "input_col:" << std::endl << input_col << std::endl;
  // Perform the matrix multiplication
  std::cout << "weight shape:" << weight_col.dimension(0) << "  "
            << weight_col.dimension(1) << std::endl;
  std::cout << "col shape:" << input_col.dimension(0) << " "
            << input_col.dimension(1) << std::endl;
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