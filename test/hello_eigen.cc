#include <Eigen/Dense>
#include <iostream>

using Eigen::Matrix;

int main() {
  Matrix<unsigned char, 2, 2> m(2, 2);
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;
  std::cout << m << std::endl;
}