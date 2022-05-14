#pragma once
#include <cmath>
#include "ActivationFunction.h"


class Relu : public ActivationFunction {
public:
    using Matrix = ActivationFunction::Matrix;
    using Vector = ActivationFunction::Vector;

    double relu(double x) {
        if (x > 0) {
            return x;
        } else {
            return 0.01 * x;
        }
    }

    double relu_derivative(double x) {
        if (x > 0) {
            return 1;
        } else {
            return 0.01;
        }
    }

    Matrix evaluate(const Matrix& x) override {
        Matrix ans = x;
        for (size_t i = 0; i < ans.rows(); ++i) {
            for (size_t j = 0; j < ans.cols(); ++j) {
                ans(i, j) = relu(x(i, j));
            }
        }

        return ans;
    }

    Matrix derivative(const Vector& x) override {
        Vector ans = x;
        for (size_t i = 0; i < ans.rows(); ++i) {
            for (size_t j = 0; j < ans.cols(); ++j) {
                ans(i, j) = relu_derivative(x(i, j));
            }
        }

        return ans.asDiagonal();
    }
};
