#pragma once
#include <Eigen/Core>

struct MatrixSizeType {
    Eigen::Index height;
    Eigen::Index width;
};

class ComputeBlock {
public:
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using StepType = double;

    ComputeBlock() = default;

    explicit ComputeBlock(MatrixSizeType shape);

    Vector evaluate_1d(const Vector &x) {
        last_input_ = x;
        last_output_ = A_ * x + b_;
        return last_output_;
    }

    Matrix evaluate_2d(const Matrix &x) const {
        return A_ * x + b_ * Matrix::Constant(1, x.cols(), 1);
    }

    Vector predict_1d(const Vector &x);

    Matrix predict_2d(const Matrix &x) const;

    void train(StepType step);

    void calculate_shift(const Matrix &chain_rule);

private:
    const Matrix &grad_A();

    const Vector &grad_b();

    const Vector &grad_x();

    Matrix A_;
    Vector b_;
    Matrix A_shift_;
    Vector b_shift_;
    MatrixSizeType shape_;
    Vector last_input_;
    Vector last_output_;
    ComputeBlock *next_ = nullptr;
    ComputeBlock *previous_ = nullptr;
    bool is_end_ = false;
    bool is_begin_ = false;

    friend class Net;
};
