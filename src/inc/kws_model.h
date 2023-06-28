#ifndef _KWS_MODEL_H_
#define _KWS_MODEL_H_
#include "Eigen/Dense"

using Eigen::Dynamic;
using Eigen::Matrix;

class KwsModel {
public:
  template <typename T> KwsModel(const T *model_data);
  ~KwsModel();
  template <typename T>
  Matrix<T, Dynamic, Dynamic> forward(const Matrix<T, Dynamic, Dynamic> &input);

private:
  void *self_data;
};
#endif