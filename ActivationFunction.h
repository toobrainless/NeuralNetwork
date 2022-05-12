#pragma once
#include <Eigen/Core>

class ActivationFunction {
public:
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    virtual Matrix evaluate(const Matrix& x) = 0;

    virtual Matrix derivative(const Vector& x) = 0;

    virtual ~ActivationFunction() {
    };
};