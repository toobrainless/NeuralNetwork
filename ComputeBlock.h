#pragma once
#include "Sigmoid.h"
#include <Eigen/Core>
#include <iostream>

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
        output_ = evaluate_1d(x);
        return output_;
    }

    void update_parameters(LearningRateType lr) {
        A_ -= lr * dA_;
        b_ -= lr * db_;
    }

    void reset_grad() {
        dA_.setZero();
        db_.setZero();
    }

    Matrix push_back(const Matrix &chain_rule);

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
//        std::cerr << "grad_A" << '\n';
//        std::cerr << "[output_] " << "rows = " << output_.rows() << " cols =  " << output_.cols() << '\n';
//        std::cerr << "[chain_rule] " << "rows = " << chain_rule.rows() << " cols =  " << chain_rule.cols() << '\n';
//        std::cerr << "[output_] " << "rows = " << output_.rows() << " cols =  " << output_.cols() << '\n';
//        std::cerr << "[input_.transpose()] " << "rows = " << input_.transpose().rows() << " cols =  " << input_.transpose().cols() << '\n';
        return Sigmoid::derivative(output_) * chain_rule * input_.transpose();
    };

    Vector grad_b(const Vector &chain_rule) {
        return Sigmoid::derivative(output_) * chain_rule;
    }

    Vector grad_x(const Vector &chain_rule) {
        return A_.transpose() * Sigmoid::derivative(output_) * chain_rule;
    };

    Matrix A_;
    Vector b_;
    Matrix dA_;
    Vector db_;
    Vector input_;
    Vector output_;
};
