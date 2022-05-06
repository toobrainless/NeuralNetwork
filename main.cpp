#include <Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
  MatrixXd m(3, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m(i, j) = i + j;
    }
  }
  std::cout << m({0}, Eigen::placeholders::all) << std::endl;
}