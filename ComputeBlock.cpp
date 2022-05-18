#include "ComputeBlock.h"

namespace {
using Matrix = ComputeBlock::Matrix;
using Vector = ComputeBlock::Vector;
}  // namespace

ComputeBlock::ComputeBlock(Index rows, Index cols, std::string activation_function)
    : A_(Matrix::Random(rows, cols)),
      b_(Vector::Random(rows)),
      dA_(Matrix::Zero(rows, cols)),
      db_(Vector::Zero(rows)) {
    if (activation_function == "sigmoid") {
        activation_function_ = new Sigmoid;
    } else if (activation_function == "relu") {
        activation_function_ = new Relu;
    } else if (activation_function == "softmax") {
        activation_function_ = new Softmax;
    } else {
        throw "there isn't such activation function";
    }
}

Matrix ComputeBlock::back_propagate(const Matrix &chain_rule) {
    dA_ += grad_A(chain_rule);
    db_ += grad_b(chain_rule);
    return grad_x(chain_rule);
}
