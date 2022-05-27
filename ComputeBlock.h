#pragma once
#include <Eigen/Core>
#include <iostream>
#include "ActivationFunction.h"
#include "Relu.h"
#include "Sigmoid.h"
#include "Softmax.h"

class ComputeBlock {
public:
    using Vector = ActivationFunction::Vector;
    using Matrix = ActivationFunction::Matrix;
    using LearningRateType = double;
    using Index = Eigen::Index;
    using BatchSizeType = size_t;

    ComputeBlock(Index rows, Index cols, std::string activation_function);

    Vector evaluate_1d(const Vector &x) const {
        return activation_function_->evaluate(A_ * x + b_);
    }

    Matrix evaluate_2d(const Matrix &x) const {
        return activation_function_->evaluate(A_ * x + b_ * Matrix::Constant(1, x.cols(), 1));
    }

    const Vector &push_forward(const Vector &x) {
        input_ = x;
        output_ = evaluate_1d(x);
        return output_;
    }

    void update_parameters(LearningRateType lr, BatchSizeType batch_size) {
        A_ -= lr * dA_ / batch_size;
        b_ -= lr * db_ / batch_size;
    }

    void reset_grad() {
        dA_.setZero();
        db_.setZero();
    }

    Matrix back_propagate(const Matrix &chain_rule);

    Matrix get_A() {
        return A_;
    }

    Vector get_b() {
        return b_;
    }

    Index rows() const {
        return A_.rows();
    }

    Index cols() const {
        return A_.cols();
    }

private:
    Matrix grad_A(const Vector &chain_rule) {
        return activation_function_->derivative(output_) * chain_rule * input_.transpose();
    };

    Vector grad_b(const Vector &chain_rule) {
        return activation_function_->derivative(output_) * chain_rule;
    }

    Vector grad_x(const Vector &chain_rule) {
        return A_.transpose() * activation_function_->derivative(output_) * chain_rule;
    };

    Matrix A_;
    Vector b_;
    Matrix dA_;
    Vector db_;
    Vector input_;
    Vector output_;
    std::unique_ptr<ActivationFunction> activation_function_ = nullptr;
};
