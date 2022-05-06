#pragma once
#include <Eigen/Core>

class ComputeBlock {
public:
  using Matrix = Eigen::MatrixXf;
  using Vector = Eigen::VectorXf;

  Vector evaluate(const Vector &x) { return A_ * x + b_; }

  Vector predict(const Vector &x);

  Matrix grad_A();

  Vector grad_b();

  Vector grad_x();


private:
  Matrix A_;
  Vector b_;
  std::shared_ptr<ComputeBlock> next_ = nullptr;
  std::shared_ptr<ComputeBlock> previous_ = nullptr;
  bool is_end_ = false;
  bool is_begin_ = false;

  friend class Net;
};