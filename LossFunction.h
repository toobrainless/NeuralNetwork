#include <Eigen/Core>

class LossFunction {
public:
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;

  double evaluate(const Vector &z, const Vector &y) {
    return (z - y).transpose() * (z - y);
  }

  double evaluate(const Matrix &z, const Matrix &y);

  Vector grad_z(const Vector &z, const Vector &y);
};