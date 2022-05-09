#pragma once
#include "Sigmoid.h"
#include <Eigen/Core>

class ComputeBlock {
public:
    using Vector = Sigmoid::Vector;
    using Matrix = Sigmoid::Matrix;
    using LearningRateType = double;
    using Index = Eigen::Index;

    ComputeBlock(Index rows, Index cols);

    Vector evaluate_1d(const Vector &x) const {
        return Sigmoid::evaluate(A_ * x + b_);
    }

    Matrix evaluate_2d(const Matrix &x) const {
        return Sigmoid::evaluate(A_ * x + b_ * Matrix::Constant(1, x.cols(), 1));
    }

    const Vector &push_forward(const Vector &x) {
        input_ = x;
        output_ = A_ * x + b_;
        return output_;
    }

    void update_parameters(LearningRateType lr) {
        A_ -= lr * dA_;
        b_ -= lr * db_;
    }

    void reset_parameters() {
        dA_.setZero();
        db_.setZero();
    }

    Matrix push_back(const Matrix &chain_rule);

private:
    Matrix grad_A(const Matrix &chain_rule) {
        return Sigmoid::derivative(output_) * chain_rule * input_.transpose();
    };

    Vector grad_b(const Matrix &chain_rule) {
        return Sigmoid::derivative(output_) * chain_rule;
    }

    Vector grad_x(const Matrix &chain_rule) {
        return A_.transpose() * Sigmoid::derivative(output_) * chain_rule;
    };

    Matrix A_;
    Vector b_;
    Matrix dA_;
    Vector db_;
    Index cols_;
    Index rows_;
    Vector input_;
    Vector output_;
};
