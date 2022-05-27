#pragma once
#include <cmath>
#include "ActivationFunction.h"

class Softmax : public ActivationFunction {
public:
    using Matrix = ActivationFunction::Matrix;
    using Vector = ActivationFunction::Vector;

    Matrix evaluate(const Matrix& x) override {
        return (exp(x.array()) / exp(x.array()).sum()).matrix();
    }

    Matrix derivative(const Vector& x) override {
        size_t n = x.rows();
        Matrix ans(n, n);
        Vector eval = evaluate(x);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i == j) {
                    ans(i, j) = eval(i) * (1 - eval(j));
                } else {
                    ans(i, j) = -eval(i) * eval(j);
                }
            }
        }

        return ans;
    }
};
