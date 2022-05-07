#include <Eigen/Dense>
#include <iostream>
#include <list>

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
    MatrixXd m = MatrixXd::Constant(1, 5, 1);
    std::cout << m << '\n';
    std::cout << m.rows() << '\n';
    std::cout << m.cols() << '\n';
}
