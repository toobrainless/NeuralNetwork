#include <Net.h>
#include <memory>;

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

Net::Net(CountType layers) : layers_(layers) {
    begin_ = std::make_shared<ComputeBlock>();
    begin_->is_begin_ = true;
    std::shared_ptr<ComputeBlock> cur = begin_;
    for (size_t i = ; i + 1 < layers; ++i) {
        cur->next_ = std::make_shared<ComputeBlock>();
        cur->next_->previous_ = cur;
        cur = cur->next_;
    }
    cur->is_end_ = true;
}