#pragma once
#include "ComputeBlock.h"
#include "LossFunction.h"
#include <vector>
#include <Eigen/Core>
#include <memory>

class Net {
public:
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using CountType = size_t;
    using TolerenceType = double;
    using StepType = double;
    using LayersShapes = std::vector<MatrixSizeType>;

    Net(const LayersShapes &layers, TolerenceType tol, StepType step);

    void feed(const Matrix &x, const Matrix &y);

    Vector predict_1d(const Vector &x) {
        return begin_->predict_1d(x);
    }

    Matrix predict_2d(const Matrix &x) {
        return begin_->predict_2d(x);
    }

    ~Net();

private:
    ComputeBlock *begin_;
    ComputeBlock *end_;
    LayersShapes layers_;
    StepType step_ = 350;
    TolerenceType tol_ = 1e-2;
    LossFunction loss_;
};
