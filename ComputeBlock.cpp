#include "ComputeBlock.h"

namespace {
using Matrix = ComputeBlock::Matrix;
using Vector = ComputeBlock::Vector;
}  // namespace

ComputeBlock::ComputeBlock(MatrixSizeType shape) : shape_(shape) {
    A_ = Matrix::Random(shape_.height, shape_.width);
    b_ = Vector::Random(shape_.height);
}

Vector ComputeBlock::predict(const Vector &x) const {
    Vector ans = evaluate(x);
    if (is_end_) {
        return ans;
    }
    return next_->predict(ans);
}

void ComputeBlock::train(const Matrix &chain_rule, StepType step) {
    A_ -= step * chain_rule * grad_A();
    b_ -= step * chain_rule * grad_b();
    if (!is_end_) {
        previous_->train(chain_rule * grad_x(), step);
    }
}
