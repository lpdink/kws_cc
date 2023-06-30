#include "ops/conv2d.hpp"
#include "utils.hpp"
using namespace Eigen;
using namespace KwsBackend;
using namespace std;

int main(){
    float *model_data = load_file<float>("tensor.bin");
    float *input_data = load_file<float>("input.bin");
    // in_channels=4, out_channels=10, kernel_size=(5, 3), stride=(1, 2), padding=(2, 1), bias=True
    int offset = 0;
    auto *layer = new Conv2D<float>(4, 10, TwoDim(5, 3), TwoDim(1, 2), TwoDim(2, 1), model_data, offset);
    // auto *input = new Map<MatrixXf>()
    // layer->forward()

}