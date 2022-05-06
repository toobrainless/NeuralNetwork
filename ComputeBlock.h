#pragma once
#include <Eigen/Core>

class ComputeBlock {
public:
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;
  using StepType = double;

  Vector evaluate(const Vector &x) { return A_ * x + b_; }

  Vector predict(const Vector &x);

  void train(const Matrix &chain_rule, StepType step);

  const Matrix &grad_A();

  const Vector &grad_b();

  const Vector &grad_x();

private:
  Matrix A_;
  Vector b_;
  Vector current_value_;
  std::shared_ptr<ComputeBlock> next_ = nullptr;
  std::shared_ptr<ComputeBlock> previous_ = nullptr;
  bool is_end_ = false;
  bool is_begin_ = false;

  friend class Net;
};