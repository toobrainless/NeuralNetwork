#include "ComputeBlock.h"

namespace {
using Matrix = ComputeBlock::Matrix;
using Vector = ComputeBlock::Vector;
}  // namespace

ComputeBlock::ComputeBlock(Index rows, Index cols)
    : A_(Matrix::Random(rows, cols)),
      b_(Vector::Random(rows)),
      dA_(Matrix::Zero(rows, cols)),
      db_(Vector::Zero(rows)) {
}

Matrix ComputeBlock::push_back(const Matrix &chain_rule) {
    dA_ += grad_A(chain_rule);
    db_ += grad_b(chain_rule);

    return grad_x(chain_rule);
}
