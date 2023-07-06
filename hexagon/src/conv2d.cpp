// #include "qurt.h"
#include "ops/conv2d.hpp"

#include <iostream>

#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "remote.h"
#include "speech_backend.h"
#include "utils.hpp"
#include "verify.h"

using namespace Eigen;
using namespace SpeechBackend;
using namespace SpeechBackend::Ops;
using namespace std;

AEEResult speech_backend_open(const char *uri, remote_handle64 *h) {
  *h = 0x00DEAD00;
  return 0;
}

AEEResult speech_backend_close(remote_handle64 h) { return 0; }

AEEResult speech_backend_plus(remote_handle64 h, const int num1, const int num2,
                              int *rst) {
  *rst = num1 + num2;
  FARF(ALWAYS, "speech_backend plus rst:%d", *rst);
  return 0;
}

AEEResult speech_backend_conv2d(remote_handle64 h) {
  int model_size = 0, input_size = 0;
  float *model_data = load_file<float>("tensor.bin", model_size);
  float *input_data = load_file<float>("input.bin", input_size);
  int offset = 0;
  auto *layer = new Conv2D<float>(4, 10, TwoDim(5, 3), TwoDim(1, 2),
                                  TwoDim(2, 1), model_data, offset);
  auto *input = new TensorMap<Tensor<float, 4, 1>>(input_data, 1, 4, 10, 10);
  auto rst = layer->forward(*input);
  int rst_batch_size = rst.dimension(0);
  int rst_channel = rst.dimension(1);
  int rst_width = rst.dimension(2);
  int rst_height = rst.dimension(3);
  write_out("cc_rst.bin", rst.data(),
            rst_batch_size * rst_channel * rst_width * rst_height);
    return 0;
}