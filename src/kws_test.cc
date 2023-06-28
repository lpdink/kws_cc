#include "utils.h"
#include <stdio.h>

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    printf("expected 2 args: model.bin input.bin, got %d", argc);
    return 0;
  }
  const char *model_path = argv[1];
  const char *input_path = argv[2];
  // TODO: test load_file;
  float *model_data = load_file(model_path);
  float *input_data = load_file(input_path);
}