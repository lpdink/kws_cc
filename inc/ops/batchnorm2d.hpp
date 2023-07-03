#ifndef _BATCHNORM2D_HPP_
#define _BATCHNORM2D_HPP_
#include "Eigen/Dense"

using Eigen::Dynamic;
using Eigen::Matrix;

namespace SpeechBackend {
namespace Ops {
template <typename T>
class BatchNorm2D {
 public:
  BatchNorm2D(int num_features);
  {
    // 初始化缩放因子和偏移量
    scale_ = Eigen::Matrix<T, Eigen::Dynamic, 1>::Ones(num_features);
    offset_ = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(num_features);
  }

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> forward(
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& input) {
    int batch_size = input.rows();
    int num_channels = input.cols();

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> output(batch_size,
                                                            num_channels);

    for (int i = 0; i < batch_size; i++) {
      Eigen::Matrix<T, Eigen::Dynamic, 1> input_row =
          input.row(i);  // 获取输入矩阵的一行数据

      // 计算均值和方差
      T mean = input_row.mean();
      T variance =
          (input_row.array() - mean).matrix().squaredNorm() / num_features_;

      // 归一化
      Eigen::Matrix<T, Eigen::Dynamic, 1> normalized =
          (input_row.array() - mean) / std::sqrt(variance + epsilon_);

      // 缩放和平移
      Eigen::Matrix<T, Eigen::Dynamic, 1> scaled =
          scale_.asDiagonal() * normalized + offset_;

      output.row(i) = scaled;
    }

    return output;
  }

 private:
  int num_features_;  // 特征数量

  Eigen::Matrix<T, Eigen::Dynamic, 1> scale_;   // 缩放因子
  Eigen::Matrix<T, Eigen::Dynamic, 1> offset_;  // 偏移量

  T epsilon_ = 1e-5;  // 防止分母为零的小常数
};

template <typename T>
BatchNorm2D<T>::BatchNorm2D(int num_features) : num_features_(num_features) {}
}  // namespace Ops
}  // namespace SpeechBackend
#endif