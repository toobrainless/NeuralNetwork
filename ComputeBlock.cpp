#include "ComputeBlock.h"

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

Vector ComputeBlock::predict(const Vector &x) {
  Vector ans = evaluate(x);
  if (is_end_) {
    return ans;
  }
  return next_->predict(ans);
}