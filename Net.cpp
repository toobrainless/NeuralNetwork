#include <Net.h>

namespace {
using Matrix = Net::Matrix;
using Vector = Net::Vector;
}

Net::Net(const LayersShapes &layers, TolerenceType tol, StepType step)
    : layers_(layers), tol_(tol), step_(step) {
    assert(!empty(layers) && "Layers must contain at least one element");
    begin_ = new ComputeBlock(layers[0]);
    begin_->is_begin_ = true;
    ComputeBlock *cur = begin_;
    for (size_t i = 1; i < layers.size(); ++i) {
        assert(layers[i - 1].height == layers[i].width && "Incorrect dimensions");
        cur->next_ = new ComputeBlock(layers[i]);
        cur->next_->previous_ = cur;
        cur = cur->next_;
    }
    cur->is_end_ = true;
}

// В этих матрицах по столбцам лежат x_i -> y_i
void Net::feed(const Matrix &x, const Matrix &y) {
    while (loss_.evaluate_2d(predict_2d(x), y) > tol_) {
        for (size_t i = 0; i < x.cols(); ++i) {
            Vector z = predict_1d(x(Eigen::all, i));
            end_->calculate_shift(loss_.grad_z(z, y(Eigen::all, i)));
        }

        end_->train(step_);
    }
}

Net::~Net() {
    ComputeBlock *cur = begin_;
    ComputeBlock *tmp;
    while (cur) {
        tmp = cur;
        cur = cur->next_;
        delete tmp;
    }
}