#pragma once
#include <cmath>
#include <Eigen/Core>

class Sigmoid {
public:
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    static Matrix evaluate(const Matrix& x) {
        return 1 / (1 + exp(-x.array()));
    }

    static Matrix derivative(const Vector& x) {
        return ((exp(-x.array())) / pow(exp(-x.array()) + 1, 2)).matrix().asDiagonal();
    }
};
