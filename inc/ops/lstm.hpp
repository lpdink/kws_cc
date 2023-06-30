#ifndef _LSTM_HPP_
#define _LSTM_HPP_
#include "Eigen/Dense"

using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;

namespace KwsBackend {

template <typename T>
struct LSTMData {
 public:
  LSTMData(int input_size, int hidden_size, T *model_data, int &offset) {
    this->w_ih = new Map<Matrix<T, Dynamic, Dynamic>>(
        model_data + offset, 4 * hidden_size, input_size);
    offset += 4 * hidden_size * input_size;
    this->w_hh = new Map<Matrix<T, Dynamic, Dynamic>>(
        model_data + offset, 4 * hidden_size, input_size);
    offset += 4 * hidden_size * input_size;
    this->b_ih =
        new Map<Matrix<T, Dynamic, 1>>(model_data + offset, 4 * hidden_size);
    offset += 4 * hidden_size;
    this->b_hh =
        new Map<Matrix<T, Dynamic, 1>>(model_data + offset, 4 * hidden_size);
    offset += 4 * hidden_size;
  }
  ~LSTMData() {
    delete this->w_ih;
    this->w_ih = nullptr;
    delete this->w_hh;
    this->w_hh = nullptr;
    delete this->b_ih;
    this->b_ih = nullptr;
    delete this->b_hh;
    this->b_hh = nullptr;
  }

 private:
  Map<Matrix<T, Dynamic, Dynamic>> *w_ih;
  Map<Matrix<T, Dynamic, Dynamic>> *w_hh;
  Map<Matrix<T, Dynamic, 1>> *b_ih;
  Map<Matrix<T, Dynamic, 1>> *b_hh;
};

template <typename T>
class LSTM {
 public:
  LSTM(int input_size, int hidden_size, T *model_data, int &offset);
  ~LSTM();

  Eigen::Matrix<T, Eigen::Dynamic, 1> forward(
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &input,
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &hidden);

 private:
  int input_size_;   // 输入维度
  int hidden_size_;  // 隐藏状态维度
  LSTMData<T> *self_data;
};

template <typename T>
LSTM<T>::LSTM(int input_size, int hidden_size, T *model_data, int &offset)
    : input_size_(input_size), hidden_size_(hidden_size) {
  this->self_data =
      new LSTMData<T>(input_size, hidden_size, model_data, offset);
}
template <typename T>
LSTM<T>::~LSTM() {
  delete this->self_data;
  this->self_data = nullptr;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> LSTM<T>::forward(
    const Eigen::Matrix<T, Eigen::Dynamic, 1> &input,
    const Eigen::Matrix<T, Eigen::Dynamic, 1> &hidden) {}
}  // namespace KwsBackend
#endif