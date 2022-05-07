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

    Vector evaluate(const Vector &x) const {
        return A_ * x + b_;
    }

    Vector predict(const Vector &x) const;

    void train(const Matrix &chain_rule, StepType step);

private:
    const Matrix &grad_A();

    const Vector &grad_b();

    const Vector &grad_x();

    Matrix A_;
    Vector b_;
    MatrixSizeType shape_;
    Vector current_value_;
    ComputeBlock *next_ = nullptr;
    ComputeBlock *previous_ = nullptr;
    bool is_end_ = false;
    bool is_begin_ = false;

    friend class Net;
};
