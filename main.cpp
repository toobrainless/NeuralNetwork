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
    MatrixXd m = VectorXd::Random(2);
    std::cout << m << '\n';
}
