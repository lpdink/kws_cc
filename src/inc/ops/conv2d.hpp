#ifndef _CONV2D_HPP_
#define _CONV2D_HPP_
#include "Eigen/Dense"

using Eigen::Dynamic;
using Eigen::Matrix;

namespace KwsBackend {
template <typename T>
class Conv2D {
 public:
  Conv2D(int input_channels, int output_channels, int kernel_size, int padding,
         int stride, const T &model_data, int &offset)
      : input_channels_(input_channels),
        output_channels_(output_channels),
        kernel_size_(kernel_size),
        padding_(padding),
        stride_(stride) {
    // 初始化卷积核权重矩阵和偏置向量
    W_ = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(
        output_channels, input_channels * kernel_size * kernel_size);
    b_ = Eigen::Matrix<T, Eigen::Dynamic, 1>::Random(output_channels);
  }

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> forward(
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &input) {
    int input_height = input.rows();
    int input_width = input.cols();

    int output_height =
        (input_height - kernel_size_ + 2 * padding_) / stride_ + 1;
    int output_width =
        (input_width - kernel_size_ + 2 * padding_) / stride_ + 1;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> output(output_height,
                                                            output_width);

    for (int i = 0; i < output_height; i++) {
      for (int j = 0; j < output_width; j++) {
        int start_row = i * stride_ - padding_;
        int start_col = j * stride_ - padding_;
        int end_row = start_row + kernel_size_;
        int end_col = start_col + kernel_size_;

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> input_patch =
            input.block(start_row, start_col, kernel_size_, kernel_size_);
        Eigen::Matrix<T, Eigen::Dynamic, 1> input_patch_flattened =
            input_patch.reshape(input_channels_ * kernel_size_ * kernel_size_,
                                1);
        Eigen::Matrix<T, Eigen::Dynamic, 1> result =
            (W_ * input_patch_flattened + b_).unaryExpr([](T val) {
              return std::max(val, static_cast<T>(0));
            });
        output(i, j) = result.sum();
      }
    }

    return output;
  }

 private:
  int input_channels_;   // 输入通道数
  int output_channels_;  // 输出通道数
  int kernel_size_;      // 卷积核大小
  int padding_;          // padding大小
  int stride_;           // 步长

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> W_;  // 卷积核权重矩阵
  Eigen::Matrix<T, Eigen::Dynamic, 1> b_;               // 偏置向量
};
}  // namespace KwsBackend
#endif