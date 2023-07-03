#ifndef _CRN_MODEL_HPP_
#define _CRN_MODEL_HPP_
#include <stdlib.h>
#include <string.h>

#include "Eigen/Dense"
#include "modules/speech_conv.hpp"
#include "modules/speech_deconv.hpp"
#include "ops/lstm.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;
using SpeechBackend::Module::SpeechConv;
using SpeechBackend::Module::SpeechDeConv;
using SpeechBackend::Ops::LSTM;

namespace SpeechBackend {
namespace Model {
namespace Crn {

template <typename T>
struct CrnModelData {
  /* data */
  SpeechConv<T> *mag_conv;
  SpeechConv<T> *angle_conv;
  SpeechConv<T> *conv2;
  SpeechConv<T> *conv3;
  SpeechConv<T> *conv4;
  SpeechConv<T> *conv5;
  SpeechConv<T> *conv6;
  SpeechConv<T> *conv7;
  LSTM<T> *lstm1;
  LSTM<T> *lstm2;
  SpeechDeConv<T> *conv1_t;
  SpeechDeConv<T> *conv2_t;
  SpeechDeConv<T> *conv3_t;
  SpeechDeConv<T> *conv4_t;
  SpeechDeConv<T> *conv5_t;
  SpeechDeConv<T> *conv6_t;
  SpeechDeConv<T> *conv7_t;
  SpeechConv<T> *conv_mag_out;
  SpeechConv<T> *conv_mask_out;
  CrnModelData(T *model_data, int &offset) {
    this->mag_conv = new SpeechConv<T>(4, 10, TwoDim(5, 3), TwoDim(1, 2),
                                       TwoDim(2, 1), model_data, offset);
    this->angle_conv = new SpeechConv<T>(4, 10, TwoDim(5, 3), TwoDim(1, 2),
                                         TwoDim(2, 1), model_data, offset);
    this->conv2 = new SpeechConv<T>(20, 20, TwoDim(1, 3), TwoDim(1, 2),
                                    TwoDim(0, 1), model_data, offset);
    this->conv3 = new SpeechConv<T>(20, 20, TwoDim(1, 3), TwoDim(1, 2),
                                    TwoDim(0, 1), model_data, offset);
    this->conv4 = new SpeechConv<T>(20, 20, TwoDim(1, 3), TwoDim(1, 2),
                                    TwoDim(0, 1), model_data, offset);
    this->conv5 = new SpeechConv<T>(20, 20, TwoDim(1, 3), TwoDim(1, 2),
                                    TwoDim(0, 1), model_data, offset);
    this->conv6 = new SpeechConv<T>(20, 20, TwoDim(1, 3), TwoDim(1, 2),
                                    TwoDim(0, 1), model_data, offset);
    this->conv7 = new SpeechConv<T>(20, 20, TwoDim(1, 3), TwoDim(1, 2),
                                    TwoDim(0, 1), model_data, offset);

    this->lstm1 = new LSTM<T>(60, 60, model_data, offset);
    this->lstm2 = new LSTM<T>(60, 60, model_data, offset);
    this->conv7_t = new SpeechDeConv<T>(40, 20, TwoDim(1, 3), TwoDim(1, 2),
                                        TwoDim(0, 1), model_data, offset);
    this->conv6_t = new SpeechDeConv<T>(40, 20, TwoDim(1, 3), TwoDim(1, 2),
                                        TwoDim(0, 1), model_data, offset);
    this->conv5_t = new SpeechDeConv<T>(40, 20, TwoDim(1, 3), TwoDim(1, 2),
                                        TwoDim(0, 1), model_data, offset);
    this->conv4_t = new SpeechDeConv<T>(40, 20, TwoDim(1, 3), TwoDim(1, 2),
                                        TwoDim(0, 1), model_data, offset);
    this->conv3_t = new SpeechDeConv<T>(40, 20, TwoDim(1, 3), TwoDim(1, 2),
                                        TwoDim(0, 1), model_data, offset);
    this->conv2_t = new SpeechDeConv<T>(40, 20, TwoDim(1, 3), TwoDim(1, 2),
                                        TwoDim(0, 1), model_data, offset);
    this->conv1_t = new SpeechDeConv<T>(40, 16, TwoDim(1, 3), TwoDim(1, 2),
                                        TwoDim(0, 1), model_data, offset);
    this->conv_mag_out = new SpeechConv<T>(20, 4, TwoDim(1, 3), TwoDim(1, 1),
                                           TwoDim(0, 1), model_data, offset);
    this->conv_mask_out = new SpeechConv<T>(20, 12, TwoDim(1, 3), TwoDim(1, 1),
                                            TwoDim(0, 1), model_data, offset);
  }
  ~CrnModelData() {
    delete this->mag_conv;
    this->mag_conv = nullptr;
    delete this->angle_conv;
    this->angle_conv = nullptr;
    delete this->conv2;
    this->conv2 = nullptr;
    delete this->conv3;
    this->conv3 = nullptr;
    delete this->conv4;
    this->conv4 = nullptr;
    delete this->conv5;
    this->conv5 = nullptr;
    delete this->conv6;
    this->conv6 = nullptr;
    delete this->conv7;
    this->conv7 = nullptr;
    delete this->lstm1;
    this->lstm1 = nullptr;
    delete this->lstm2;
    this->lstm2 = nullptr;
    delete this->conv7_t;
    this->conv7_t = nullptr;
    delete this->conv6_t;
    this->conv6_t = nullptr;
    delete this->conv5_t;
    this->conv5_t = nullptr;
    delete this->conv4_t;
    this->conv4_t = nullptr;
    delete this->conv3_t;
    this->conv3_t = nullptr;
    delete this->conv2_t;
    this->conv2_t = nullptr;
    delete this->conv1_t;
    this->conv1_t = nullptr;
    delete this->conv_mag_out;
    this->conv_mag_out = nullptr;
    delete this->conv_mask_out;
    this->conv_mask_out = nullptr;
  }
};

template <typename T>
class CrnModel {
 public:
  CrnModel(T *model_data);
  ~CrnModel();
  Matrix<T, Dynamic, Dynamic> forward(const Matrix<T, Dynamic, Dynamic> &input);

 private:
  CrnModelData<T> *self_data;
};

template <typename T>
CrnModel<T>::CrnModel(T *model_data) {
  int offset = 0;
  CrnModelData<T> *data = new CrnModelData<T>(model_data, offset);
  memset(data, 0, sizeof(CrnModelData<T>));
  if (data == nullptr) {
    printf("CrnModel allocate memory failed.");
    exit(0);
  }

  this->self_data = data;
  printf("\noffset:%d\n", offset);
}

template <typename T>
CrnModel<T>::~CrnModel() {
  delete this->self_data;
  this->self_data = nullptr;
}

template <typename T>
Matrix<T, Dynamic, Dynamic> CrnModel<T>::forward(
    const Matrix<T, Dynamic, Dynamic> &input) {
  printf("\nModel Forward Not Implemented\n");
  Matrix<T, Dynamic, Dynamic> tmp = Matrix<T, Dynamic, Dynamic>(0, 0);
  return tmp;
}
}  // namespace Crn
}  // namespace Model
}  // namespace SpeechBackend
#endif