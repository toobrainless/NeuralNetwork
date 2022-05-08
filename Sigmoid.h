#pragma once
#include <cmath>
#include <Eigen/Core>

class Sigmoid {
public:
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    static Vector evaluate(const Vector& x) {
        return 1 / (1 + exp(-x.array()));
    }
};
