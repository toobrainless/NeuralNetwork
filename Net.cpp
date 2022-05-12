#include <Net.h>

namespace {
using Matrix = Net::Matrix;
using Vector = Net::Vector;
using Index = Net::Index;
using StepType = Net::LearningRateType;
}  // namespace

Net::Net(const std::vector<Index>& layers_sizes, std::string activation_function, TolerenceType tol, LearningRateType lr)
    : tol_(tol), lr_(lr) {
    if (activation_function == "sigmoid") {
        activation_function_ = new Sigmoid;
    } else {
        throw "There isn't such activation function";
    }
    assert(layers_sizes.size() >= 2);
    layers_.reserve(layers_sizes.size() - 1);
    for (size_t i = 0; i + 1 < layers_sizes.size(); ++i) {
        layers_.emplace_back(layers_sizes[i + 1], layers_sizes[i], activation_function_);
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
    double error;
    size_t k = 0;
    while ((error = loss_.evaluate_2d(predict_2d(x), y)) > tol_) {
        reset_grad();
        for (size_t i = 0; i < x.cols(); ++i) {
            z = push_forward(x(Eigen::all, i));
            push_back(z, y(Eigen::all, i));
        }

        if (k == 0) {
            std::cout << error << '\n';
        }

        ++k;

        if (k % 100000 == 0) {
            k = 1;
            std::cout << error << '\n';
        }

        update_parameters(lr_);
//        reset_grad();
    }

    std::cout << error << '\n';
}

Vector Net::push_forward(const Vector& x) {
    Vector arg = x;
    for (auto& layer : layers_) {
        arg = layer.push_forward(arg);
    }

    return arg;
};

void Net::push_back(const Vector& z, const Vector& y) {
    Matrix arg = loss_.grad_z(z, y);
    reverse(layers_.begin(), layers_.end());
    for (auto& layer : layers_) {
        arg = layer.push_back(arg);
    }
    reverse(layers_.begin(), layers_.end());
};

void Net::update_parameters(LearningRateType lr) {
    for (auto& layer : layers_) {
        layer.update_parameters(lr);
    }
}

void Net::reset_grad() {
    for (auto& layer : layers_) {
        layer.reset_grad();
    }
}
