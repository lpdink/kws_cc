#include "ops/conv2d.hpp"

#include <iostream>

#include "utils.hpp"
using namespace Eigen;
using namespace SpeechBackend;
using namespace SpeechBackend::Ops;
using namespace std;

int main() {
  int model_size = 0, input_size = 0;
  float *model_data = load_file<float>("tensor.bin", model_size);
  float *input_data = load_file<float>("input.bin", input_size);
  // in_channels=4, out_channels=10, kernel_size=(5, 3), stride=(1, 2),
  // padding=(2, 1), bias=True
  int offset = 0;
  auto *layer = new Conv2D<float>(4, 10, TwoDim(5, 3), TwoDim(1, 2),
                                  TwoDim(2, 1), model_data, offset);
  cout << offset << endl;
  auto *input = new TensorMap<Tensor<float, 4, 1>>(input_data, 1, 4, 10, 10);
  auto rst = layer->forward(*input);
  int rst_batch_size = rst.dimension(0);
  int rst_channel = rst.dimension(1);
  int rst_width = rst.dimension(2);
  int rst_height = rst.dimension(3);
  //   float * ptr = rst.data();
  //   fopen()
  cout << "--------------output shape:-------------" << endl;
  cout << rst_batch_size << " " << rst_channel << " " << rst_width << " "
       << rst_height << endl;
  write_out("cc_rst.bin", rst.data(),
            rst_batch_size * rst_channel * rst_width * rst_height);
}