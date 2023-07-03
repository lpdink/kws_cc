#include <stdio.h>

#include "models/crn_model.hpp"
#include "utils.hpp"
using namespace SpeechBackend;
using namespace SpeechBackend::Model::Crn;

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    printf("expected 2 args: model.bin input.bin, got %d\n", argc);
    return 0;
  }
  const char *model_path = argv[1];
  const char *input_path = argv[2];
  int model_size = 0, input_size = 0;
  float *model_data = load_file<float>(model_path, model_size);
  float *input_data = load_file<float>(input_path, input_size);
  CrnModel<float> *model = new CrnModel<float>(model_data);
  printf("\nmodel size:%d\n", model_size);
  Map<Eigen::VectorXf> input(input_data, input_size);
  model->forward(input);
}