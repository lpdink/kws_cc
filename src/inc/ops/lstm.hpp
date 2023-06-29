#ifndef _LSTM_HPP_
#define _LSTM_HPP_
#include "Eigen/Dense"

using Eigen::Dynamic;
using Eigen::Matrix;

namespace KwsBackend {
template <typename T>
class LSTM {
 public:
  LSTM(int input_size, int hidden_size, const T &model_data, int &offset)
      : input_size_(input_size), hidden_size_(hidden_size) {
    // 初始化权重矩阵和偏置向量
    Wf_ = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(
        hidden_size, input_size + hidden_size);
    bf_ = Eigen::Matrix<T, Eigen::Dynamic, 1>::Random(hidden_size);
    Wi_ = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(
        hidden_size, input_size + hidden_size);
    bi_ = Eigen::Matrix<T, Eigen::Dynamic, 1>::Random(hidden_size);
    Wo_ = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(
        hidden_size, input_size + hidden_size);
    bo_ = Eigen::Matrix<T, Eigen::Dynamic, 1>::Random(hidden_size);
    Wc_ = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(
        hidden_size, input_size + hidden_size);
    bc_ = Eigen::Matrix<T, Eigen::Dynamic, 1>::Random(hidden_size);
  }

  Eigen::Matrix<T, Eigen::Dynamic, 1> forward(
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &input) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> h_prev =
        Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(
            hidden_size_);  // 上一个时刻的隐藏状态
    Eigen::Matrix<T, Eigen::Dynamic, 1> c_prev =
        Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(
            hidden_size_);  // 上一个时刻的细胞状态

    Eigen::Matrix<T, Eigen::Dynamic, 1> x(input_size_ + hidden_size_);
    x << input, h_prev;

    // 计算遗忘门
    Eigen::Matrix<T, Eigen::Dynamic, 1> f =
        (Wf_ * x + bf_).unaryExpr([](T val) {
          return 1 / (1 + std::exp(-val));
        });

    // 计算输入门
    Eigen::Matrix<T, Eigen::Dynamic, 1> i =
        (Wi_ * x + bi_).unaryExpr([](T val) {
          return 1 / (1 + std::exp(-val));
        });

    // 计算细胞状态的更新
    Eigen::Matrix<T, Eigen::Dynamic, 1> c =
        (Wc_ * x + bc_).unaryExpr([](T val) { return std::tanh(val); });
    c = f.cwiseProduct(c_prev) + i.cwiseProduct(c);

    // 计算输出门
    Eigen::Matrix<T, Eigen::Dynamic, 1> o =
        (Wo_ * x + bo_).unaryExpr([](T val) {
          return 1 / (1 + std::exp(-val));
        });

    // 计算隐藏状态的更新
    Eigen::Matrix<T, Eigen::Dynamic, 1> h =
        o.cwiseProduct(c.unaryExpr([](T val) { return std::tanh(val); }));

    // 更新上一个时刻的隐藏状态和细胞状态
    h_prev = h;
    c_prev = c;

    return h;
  }

 private:
  int input_size_;   // 输入维度
  int hidden_size_;  // 隐藏状态维度

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Wf_;  // 遗忘门权重矩阵
  Eigen::Matrix<T, Eigen::Dynamic, 1> bf_;  // 遗忘门偏置向量
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Wi_;  // 输入门权重矩阵
  Eigen::Matrix<T, Eigen::Dynamic, 1> bi_;  // 输入门偏置向量
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Wo_;  // 输出门权重矩阵
  Eigen::Matrix<T, Eigen::Dynamic, 1> bo_;  // 输出门偏置向量
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Wc_;  // 细胞状态权重矩阵
  Eigen::Matrix<T, Eigen::Dynamic, 1> bc_;  // 细胞状态偏置向量
};
}  // namespace KwsBackend
#endif