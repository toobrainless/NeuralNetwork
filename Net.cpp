#include <Net.h>

Net::Net(CountType layers, TolerenceType tol, StepType step)
    : layers_(layers), tol_(tol), step_(step) {
  loss_ = std::make_unique<LossFunction>();
  begin_ = std::make_shared<ComputeBlock>();
  begin_->is_begin_ = true;
  std::shared_ptr<ComputeBlock> cur = begin_;
  for (size_t i = 0; i + 1 < layers; ++i) {
    cur->next_ = std::make_shared<ComputeBlock>();
    cur->next_->previous_ = cur;
    cur = cur->next_;
  }
  cur->is_end_ = true;
}

void Net::feed(const Matrix &x, const Matrix &y) {
  while (loss_->evaluate(predict(x), y) > tol_) {
    end_->train(loss_->grad_z(predict(x), y), step_);
  }
}