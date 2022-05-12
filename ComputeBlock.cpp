#include "ComputeBlock.h"

namespace {
using Matrix = ComputeBlock::Matrix;
using Vector = ComputeBlock::Vector;
}  // namespace

ComputeBlock::ComputeBlock(Index rows, Index cols, ActivationFunction* activation_function)
    : A_(Matrix::Random(rows, cols)),
      b_(Vector::Random(rows)),
      dA_(Matrix::Zero(rows, cols)),
      db_(Vector::Zero(rows)),
      activation_function_(activation_function) {

    std::cout << "ComputeBlock::A_ " << A_ << "\n\n";
    std::cout << "ComputeBlock::b_ " << b_ << "\n\n";

}

Matrix ComputeBlock::push_back(const Matrix &chain_rule) {
    dA_ += grad_A(chain_rule);
    db_ += grad_b(chain_rule);

    return grad_x(chain_rule);
}
