#include <Eigen/Dense>
#include <iostream>
#include <list>
#include <cmath>
#include "Net.h"
#include "Tests.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

int main() {
//    Matrix test(Matrix::Random(3, 3));
//    std::cout << test << '\n';
    test_all();
}
