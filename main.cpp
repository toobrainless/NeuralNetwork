#include <Eigen/Dense>
#include <iostream>
#include <list>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Tester {
public:
    Tester(double kek) : kek_(kek) {
    }
    double kek_;

    void check() {
        std::cout << (this)->kek_;
    }
};

int main() {
    MatrixXd m = MatrixXd(2, 2);
    m(0, 0) = 1;
    m(1, 0) = 5;
    m(0, 1) = 7;
    m(1, 1) = 10;
    std::cout << pow(m.array(), 2) << '\n';
    std::cout << m << '\n';
//    std::cout << m.rows() << '\n';
//    std::cout << m.cols() << '\n';
}
