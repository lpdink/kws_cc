#ifndef _KWS_MODEL_HPP_
#define _KWS_MODEL_HPP_
#include <stdlib.h>
#include <string.h>

#include "Eigen/Dense"
#include "modules/speech_conv.hpp"
#include "modules/speech_deconv.hpp"
#include "ops/lstm.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;
namespace KwsBackend {
template <typename T>
class KwsModel {
 public:
  KwsModel(const T *model_data);
  ~KwsModel();
  Matrix<T, Dynamic, Dynamic> forward(const Matrix<T, Dynamic, Dynamic> &input);

 private:
  void *self_data;
};

template <typename T>
KwsModel<T>::KwsModel(const T *model_data) {}

template <typename T>
struct KwsModelData {
  /* data */
  SpeechConv<T> *mag_conv;
  SpeechConv<T> *angle_conv;
  SpeechConv<T> *conv2;
  SpeechConv<T> *conv3;
  SpeechConv<T> *conv4;
  SpeechConv<T> *conv5;
  SpeechConv<T> *conv6;
  SpeechConv<T> *conv7;
  LSTM<T> *lstm;
  SpeechDeConv<T> *conv1_t;
  SpeechDeConv<T> *conv2_t;
  SpeechDeConv<T> *conv3_t;
  SpeechDeConv<T> *conv4_t;
  SpeechDeConv<T> *conv5_t;
  SpeechDeConv<T> *conv6_t;
  SpeechDeConv<T> *conv7_t;
  SpeechConv<T> *conv_mag_out;
  SpeechConv<T> *conv_mask_out;
};

template <typename T>
KwsModel<T>::KwsModel(const T *model_data) {
  KwsModelData<T> *data = new KwsModelData<T>();
  if (data == nullptr) {
    printf("KwsModel allocate memory failed.");
    exit(0);
  }
  memset(data, 0, sizeof(KwsModelData));
  int offset = 0;
  data->mag_conv = new SpeechConv<T>(4, 10, TwoDim(5, 3), TwoDim(1, 2),
                                     TwoDim(2, 1), true, model_data, offset);
  data->angle_conv = new SpeechConv<T>(4, 10, TwoDim(5, 3), TwoDim(1, 2),
                                       TwoDim(2, 1), true, model_data, offset);
  data->conv2 = new SpeechConv<T>(20, 20, TwoDim(1, 3), TwoDim(1, 2),
                                  TwoDim(0, 1), true, model_data, offset);
  data->conv3 = new SpeechConv<T>(20, 20, TwoDim(1, 3), TwoDim(1, 2),
                                  TwoDim(0, 1), true, model_data, offset);
  data->conv4 = new SpeechConv<T>(20, 20, TwoDim(1, 3), TwoDim(1, 2),
                                  TwoDim(0, 1), true, model_data, offset);
  data->conv5 = new SpeechConv<T>(20, 20, TwoDim(1, 3), TwoDim(1, 2),
                                  TwoDim(0, 1), true, model_data, offset);
  data->conv6 = new SpeechConv<T>(20, 20, TwoDim(1, 3), TwoDim(1, 2),
                                  TwoDim(0, 1), true, model_data, offset);
  data->conv7 = new SpeechConv<T>(20, 20, TwoDim(1, 3), TwoDim(1, 2),
                                  TwoDim(0, 1), true, model_data, offset);

  // TODO: LSTM && DeConv.

  self_data = data;
}

template <typename T>
KwsModel<T>::~KwsModel() {}

template <typename T>
Matrix<T, Dynamic, Dynamic> KwsModel<T>::forward(
    const Matrix<T, Dynamic, Dynamic> &input) {}
}  // namespace KwsBackend
#endif