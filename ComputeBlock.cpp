#include "ComputeBlock.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

Vector ComputeBlock::predict(const Vector &x) {
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