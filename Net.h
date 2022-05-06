#pragma once
#include "ComputeBlock.h"
#include "LossFunction.h"
#include <Eigen/Core>
#include <memory>

class Net {
public:
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;
  using CountType = size_t;
  using TolerenceType = double;
  using StepType = double;

  Net(CountType layers, TolerenceType tol, StepType step);

  void feed(const Matrix &x, const Matrix &y);

  Vector predict(const Vector &x) { return begin_->predict(x); }

  // Здесь хотим считать предикт сразу для матрицы в которой по строкам/столбцам
  // (надо понять как лучше) уложены наши параметры
  Matrix predict(const Matrix &X);

private:
  std::shared_ptr<ComputeBlock> begin_;
  std::shared_ptr<ComputeBlock> end_;
  CountType layers_;
  StepType step_ = 350;
  TolerenceType tol_ = 1e-2;
  std::unique_ptr<LossFunction> loss_;
};