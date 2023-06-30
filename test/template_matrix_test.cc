#include <iostream>

#include "Eigen/Dense"
using namespace Eigen;
using namespace std;

int main() {
  float *array = new float[4];
  array[0] = 1.1;
  array[1] = 2.2;
  array[2] = 3.3;
  array[3] = 4.4;

  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> *ptr =
      new Map<MatrixXf>(array + 0, 2, 2);
  cout << *ptr << endl;
}