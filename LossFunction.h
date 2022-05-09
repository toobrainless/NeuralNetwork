#include <Eigen/Core>

class LossFunction {
public:
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    double evaluate_1d(const Vector &z, const Vector &y) {
        return (z - y).transpose() * (z - y);
    }

    double evaluate_2d(const Matrix &z, const Matrix &y) {
        assert((z.cols() == y.cols() && z.rows() == y.rows()) &&
               "The shapes of the matrices must match");
        return ((z - y).transpose() * (z - y)).trace() / z.cols();
    }

    Vector grad_z(const Vector &z, const Vector &y) {
        return 2 * (z - y);
    };
};
