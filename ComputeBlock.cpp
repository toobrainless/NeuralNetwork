#include "ComputeBlock.h"

namespace {
using Matrix = ComputeBlock::Matrix;
using Vector = ComputeBlock::Vector;
}  // namespace

ComputeBlock::ComputeBlock(MatrixSizeType shape) : shape_(shape) {
    A_ = Matrix::Random(shape_.height, shape_.width);
    b_ = Vector::Random(shape_.height);
}

Vector ComputeBlock::predict_1d(const Vector &x) {
    Vector ans = evaluate_1d(x);
    if (is_end_) {
        return ans;
    }
    return next_->predict_1d(ans);
}

Matrix ComputeBlock::predict_2d(const Matrix &x) const {
    Vector ans = evaluate_2d(x);
    if (is_end_) {
        return ans;
    }
    return next_->predict_2d(ans);
}

void ComputeBlock::train(StepType step) {
    A_ -= step * A_shift_;
    b_ -= step * b_shift_;
    if (!is_begin_) {
        previous_->train(step);
    }
}

void ComputeBlock::calculate_shift(const Matrix &chain_rule) {
    A_shift_ += chain_rule * grad_A();
    b_shift_ += chain_rule * grad_b();
    if (!is_begin_) {
        previous_->calculate_shift(chain_rule * grad_x());
    }
}
