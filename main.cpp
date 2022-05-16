#include <Eigen/Dense>
#include <iostream>
#include <list>
#include <cmath>
#include "Net.h"
#include "Tests.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

int main() {
    //    Matrix test(Matrix::Random(5, 5) * 100);
    //    std::cout << test << "\n\n";

    //    std::cout << test(Eigen::all, {1, 2, 3});
    test_all();

    //    Vector v1 {{2, 1, 0.1}};
    //    std::cout << exp(v1.array())/exp(v1.array()).sum() << '\n';
}
