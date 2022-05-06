#pragma once
#include "ComputeBlock.h"
#include <Eigen/Core>

class Net {
public:
  using Matrix = Eigen::MatrixXf;
  using Vector = Eigen::VectorXf;
  using CountType = size_t;

  Net(CountType layers);

  void feed(const Matrix &x, const Vector &y);

  Vector predict(const Vector &x) { return begin_->predict(x); }

private:
  std::shared_ptr<ComputeBlock> begin_;
  std::shared_ptr<ComputeBlock> end_;
  CountType layers_;
};