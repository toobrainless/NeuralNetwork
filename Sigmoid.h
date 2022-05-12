#pragma once
#include <cmath>
#include "ActivationFunction.h"


class Sigmoid : public ActivationFunction {
public:
    using Matrix = ActivationFunction::Matrix;
    using Vector = ActivationFunction::Vector;

    Matrix evaluate(const Matrix& x) override {
        return (1 / (1 + exp(-x.array()))).matrix();
    }

    Matrix derivative(const Vector& x) override {
        return ((exp(-x.array())) / pow(exp(-x.array()) + 1, 2)).matrix().asDiagonal();
    }
};
