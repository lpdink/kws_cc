#ifndef _CONV2D_T_HPP_
#define _CONV2D_T_HPP_
#include "Eigen/Dense"

using Eigen::Dynamic;
using Eigen::Matrix;

namespace KwsBackend {

template <typename T>
class ConvTranspose2D {
 public:
  ConvTranspose2D(int input_channels, int output_channels, int kernel_size,
                  int padding, int stride, const T &model_data, int &offset)
      : input_channels_(input_channels),
        output_channels_(output_channels),
        kernel_size_(kernel_size),
        padding_(padding),
        stride_(stride) {
    // 初始化转置卷积核权重矩阵和偏置向量
    W_ = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(
        input_channels * kernel_size * kernel_size, output_channels);
    b_ = Eigen::Matrix<T, Eigen::Dynamic, 1>::Random(output_channels);
  }

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> forward(
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &input) {
    int input_height = input.rows();
    int input_width = input.cols();

    int output_height =
        (input_height - 1) * stride_ - 2 * padding_ + kernel_size_;
    int output_width =
        (input_width - 1) * stride_ - 2 * padding_ + kernel_size_;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> output(output_height,
                                                            output_width);

    for (int i = 0; i < input_height; i++) {
      for (int j = 0; j < input_width; j++) {
        Eigen::Matrix<T, Eigen::Dynamic, 1> input_patch =
            input(i, j) * Eigen::Matrix<T, Eigen::Dynamic, 1>::Ones(
                              input_channels_ * kernel_size_ * kernel_size_);
        Eigen::Matrix<T, Eigen::Dynamic, 1> result =
            (W_ * input_patch + b_).unaryExpr([](T val) {
              return std::max(val, static_cast<T>(0));
            });

        for (int m = 0; m < kernel_size_; m++) {
          for (int n = 0; n < kernel_size_; n++) {
            int output_row = i * stride_ + m - padding_;
            int output_col = j * stride_ + n - padding_;

            if (output_row >= 0 && output_row < output_height &&
                output_col >= 0 && output_col < output_width) {
              output(output_row, output_col) += result(m * kernel_size_ + n);
            }
          }
        }
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

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> W_;  // 转置卷积核权重矩阵
  Eigen::Matrix<T, Eigen::Dynamic, 1> b_;               // 偏置向量
};
}  // namespace KwsBackend
#endif