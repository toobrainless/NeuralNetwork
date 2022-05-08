#include "ComputeBlock.h"

namespace {
using Matrix = ComputeBlock::Matrix;
using Vector = ComputeBlock::Vector;
}  // namespace

ComputeBlock::ComputeBlock(Index rows, Index cols)
    : rows_(rows),
      cols_(cols),
      A_(Matrix::Random(rows, cols)),
      b_(Vector::Random(rows)),
      dA_(Matrix::Zero(rows, cols)),
      db_(Vector::Zero(rows)) {
}

Matrix ComputeBlock::push_back(const Matrix &chain_rule) {
    dA_ += chain_rule * grad_A();
    db_ += chain_rule * grad_b();

    return chain_rule * grad_x();
}
