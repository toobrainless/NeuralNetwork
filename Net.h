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
#include <random>

class Net {
public:
    using Matrix = ComputeBlock::Matrix;
    using Vector = ComputeBlock::Vector;
    using Index = ComputeBlock::Index;
    using CountType = size_t;
    using TolerenceType = double;
    using EpochType = size_t;
    using BatchSizeType = ComputeBlock::BatchSizeType;
    using LearningRateType = ComputeBlock::LearningRateType;

    Net(const std::vector<Index>& layers_sizes, const std::vector<std::string>& layers_types,
        EpochType epoch, BatchSizeType batch_size, LearningRateType lr = 1e-2);

    void train(const Matrix& x, const Matrix& y);

    Vector predict_1d(const Vector& x) const;

    Matrix predict_2d(const Matrix& x) const;

private:
    Vector push_forward(const Vector& x);
    void back_propagate(const Vector& x, const Vector& y);
    void update_parameters(LearningRateType lr);
    void reset_grad();

    void generate_batch(size_t batch_size, size_t l, size_t r, std::vector<Index>& ar);

    std::vector<ComputeBlock> layers_;
    LearningRateType lr_;
    EpochType epoch_;
    BatchSizeType batch_size_;
    LossFunction loss_;
    ActivationFunction* activation_function_ = nullptr;
};
