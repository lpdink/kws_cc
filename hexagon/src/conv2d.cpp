// #include "qurt.h"
#include "ops/conv2d.hpp"

#include <iostream>

#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "remote.h"
#include "speech_backend.h"
#include "utils.hpp"
#include "verify.h"

using namespace Eigen;
using namespace SpeechBackend;
using namespace SpeechBackend::Ops;
using namespace std;
using conv2d_int8=unsigned char;

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

AEEResult speech_backend_time_test(remote_handle64 h, int runtimes, unsigned long long *run_time_fp32, unsigned long long *run_time_int8){
    Tensor<float, 4> weight_fp32(10, 4, 5, 3);
    weight_fp32.setRandom();
    float *model_data_fp32 = weight_fp32.data();
    int offset = 0;
    auto *layer_fp32 = new Conv2D<float>(4, 10, TwoDim(5, 3), TwoDim(1, 2),
                                  TwoDim(2, 1), model_data_fp32, offset);
    auto input_fp32 = Tensor<float, 4, 1>(1, 4, 10, 10);
    input_fp32.setRandom();
    FARF(ALWAYS, "====Beigen conv2d time test for fp32====");
    unsigned long long func_start_time = HAP_perf_get_time_us();
    for(int i=0;i<runtimes;i++){
        layer_fp32->forward(input_fp32);
    }
    unsigned long long func_end_time = HAP_perf_get_time_us();
    *run_time_fp32 = func_end_time - func_start_time;
    FARF(RUNTIME_HIGH, "=============== conv2d fp32: finish. Use time: %lluus ===============", *run_time_fp32);
    delete layer_fp32;
    //=====================================
    Tensor<conv2d_int8, 4> weight_int8(10, 4, 5, 3);
    weight_int8.setRandom();
    conv2d_int8 *model_data_int8 = weight_int8.data();
    offset = 0;
    auto *layer_int8 = new Conv2D<conv2d_int8>(4, 10, TwoDim(5, 3), TwoDim(1, 2),
                                  TwoDim(2, 1), model_data_int8, offset);
    auto input_int8 = Tensor<conv2d_int8, 4, 1>(1, 4, 10, 10);
    input_int8.setRandom();
    FARF(ALWAYS, "====Beigen conv2d time test for int8====");
    func_start_time = HAP_perf_get_time_us();
    for(int i=0;i<runtimes;i++){
        layer_int8->forward(input_int8);
    }
    func_end_time = HAP_perf_get_time_us();
    *run_time_int8 = func_end_time - func_start_time;
    FARF(RUNTIME_HIGH, "=============== conv2d int8: finish. Use time: %lluus ===============", *run_time_int8);
    delete layer_int8;
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