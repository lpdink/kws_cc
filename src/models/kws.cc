#include <stdlib.h>
#include <string.h>

#include "kws_model.h"
#include "speech_conv.h"
#include "speech_deconv.h"

class SpeechConv;
class LSTM;
class SpeechDeConv;

typedef struct KwsModelData {
  /* data */
  SpeechConv *mag_conv;
  SpeechConv *angle_conv;
  SpeechConv *conv2;
  SpeechConv *conv3;
  SpeechConv *conv4;
  SpeechConv *conv5;
  SpeechConv *conv6;
  SpeechConv *conv7;
  LSTM *lstm;
  SpeechDeConv *conv1_t;
  SpeechDeConv *conv2_t;
  SpeechDeConv *conv3_t;
  SpeechDeConv *conv4_t;
  SpeechDeConv *conv5_t;
  SpeechDeConv *conv6_t;
  SpeechDeConv *conv7_t;
  SpeechConv *conv_mag_out;
  SpeechConv *conv_mask_out;
};

template <typename T>
KwsModel::KwsModel(const T *model_data) {
  KwsModelData *data = new KwsModelData();
  if (data == nullptr) {
    printf("KwsModel allocate memory failed.");
    exit(0);
  }
  memset(data, 0, sizeof(KwsModelData));
  int offset = 0;
  data->mag_conv = new SpeechConv(4, 10, TwoDim(5, 3), TwoDim(1, 2),
                                  TwoDim(2, 1), true, model_data, offset);
  data->angle_conv = new SpeechConv(4, 10, TwoDim(5, 3), TwoDim(1, 2),
                                    TwoDim(2, 1), true, model_data, offset);
  data->conv2 = new SpeechConv(20, 20, TwoDim(1, 3), TwoDim(1, 2), TwoDim(0, 1),
                               true, model_data, offset);
  data->conv3 = new SpeechConv(20, 20, TwoDim(1, 3), TwoDim(1, 2), TwoDim(0, 1),
                               true, model_data, offset);
  data->conv4 = new SpeechConv(20, 20, TwoDim(1, 3), TwoDim(1, 2), TwoDim(0, 1),
                               true, model_data, offset);
  data->conv5 = new SpeechConv(20, 20, TwoDim(1, 3), TwoDim(1, 2), TwoDim(0, 1),
                               true, model_data, offset);
  data->conv6 = new SpeechConv(20, 20, TwoDim(1, 3), TwoDim(1, 2), TwoDim(0, 1),
                               true, model_data, offset);
  data->conv7 = new SpeechConv(20, 20, TwoDim(1, 3), TwoDim(1, 2), TwoDim(0, 1),
                               true, model_data, offset);

  // TODO: LSTM && DeConv.

  self_data = data;
}

KwsModel::~KwsModel() {}