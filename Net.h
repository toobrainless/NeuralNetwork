#pragma once
#include "ComputeBlock.h"
#include "LossFunction.h"
#include <vector>
#include <Eigen/Core>
#include <memory>
#include <list>
#include <initializer_list>
#include <iostream>
#include "ActivationFunction.h"
#include "Relu.h"
#include "Sigmoid.h"

class Net {
public:
    using Matrix = ComputeBlock::Matrix;
    using Vector = ComputeBlock::Vector;
    using Index = ComputeBlock::Index;
    using CountType = size_t;
    using TolerenceType = double;
    using LearningRateType = ComputeBlock::LearningRateType;

    Net(const std::vector<Index>& layers_sizes, std::string activation_function = "sigmoid", TolerenceType tol = 1e-5, LearningRateType lr_ = 1e-2);

    void train(const Matrix& x, const Matrix& y);

    Vector predict_1d(const Vector& x) const;

    Matrix predict_2d(const Matrix& x) const;

private:
    Vector push_forward(const Vector& x);
    void push_back(const Vector& x, const Vector& y);
    void update_parameters(LearningRateType lr);
    void reset_grad();

    std::vector<ComputeBlock> layers_;
    LearningRateType lr_;
    TolerenceType tol_;
    LossFunction loss_;
    ActivationFunction* activation_function_ = nullptr;
};
