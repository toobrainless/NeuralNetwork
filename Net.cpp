#include <Net.h>

namespace {
using Matrix = Net::Matrix;
using Vector = Net::Vector;
using Index = Net::Index;
using StepType = Net::LearningRateType;
using EpochType = Net::EpochType;
}  // namespace

Net::Net(const std::initializer_list<Index>& layers_sizes,
         const std::initializer_list<std::string>& layers_types, EpochType epoch,
         BatchSizeType batch_size, LearningRateType lr)
    : epoch_(epoch), lr_(lr), batch_size_(batch_size) {
    assert(layers_sizes.size() >= 2);
    layers_.reserve(layers_sizes.size() - 1);
    for (size_t i = 0; i + 1 < layers_sizes.size(); ++i) {
        layers_.emplace_back(*(layers_sizes.begin() + (i + 1)), *(layers_sizes.begin() + i),
                             *(layers_types.begin() + i));
    }
}

Vector Net::predict_1d(const Vector& x) const {
    Vector arg = x;
    for (const auto& layer : layers_) {
        arg = layer.evaluate_1d(arg);
    }
    return arg;
}

Matrix Net::predict_2d(const Matrix& x) const {
    Matrix arg = x;
    for (const auto& layer : layers_) {
        arg = layer.evaluate_2d(arg);
    }

    return arg;
}

void Net::train(const Matrix& x, const Matrix& y) {
    Vector z;
    std::vector<Index> batch(batch_size_);
    for (EpochType j = 0; j < epoch_; ++j) {
        reset_grad();
        generate_batch(batch_size_, 0, x.cols() - 1, batch);
        for (const auto& i : batch) {
            z = push_forward(x(Eigen::all, i));
            back_propagate(z, y(Eigen::all, i));
        }

        if (j % 10 == 0) {
            std::cout << loss_.evaluate_2d(predict_2d(x), y) << '\n';
        }

        update_parameters(lr_);
    }
}

Vector Net::push_forward(const Vector& x) {
    Vector arg = x;
    for (auto& layer : layers_) {
        arg = layer.push_forward(arg);
    }

    return arg;
};

void Net::back_propagate(const Vector& z, const Vector& y) {
    Matrix arg = loss_.grad_z(z, y);
    reverse(layers_.begin(), layers_.end());
    for (auto& layer : layers_) {
        arg = layer.back_propagate(arg);
    }
    reverse(layers_.begin(), layers_.end());
};

void Net::update_parameters(LearningRateType lr) {
    for (auto& layer : layers_) {
        layer.update_parameters(lr, batch_size_);
    }
}

void Net::reset_grad() {
    for (auto& layer : layers_) {
        layer.reset_grad();
    }
}

void Net::generate_batch(size_t batch_size, size_t l, size_t r, std::vector<Index>& ar) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(l, r);
    ar.reserve(batch_size);
    for (auto& num : ar) {
        num = distrib(gen);
    }
}
