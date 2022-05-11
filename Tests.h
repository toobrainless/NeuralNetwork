#include "Net.h"

using Vector = Net::Vector;
using Matrix = Net::Matrix;

void test_loss() {
    LossFunction loss;

    {
        Vector v1{{1, 2, 3, 4}};
        Vector v2{{4, 3, 2, 1}};
        assert(loss.evaluate_1d(v1, v2) == 20);
    }

    {
        Vector v1{{1, 2, 3, 4}};
        Vector v2{{1, 2, 3, 4}};
        assert(loss.evaluate_1d(v1, v2) == 0);
    }

    {
        Matrix A {
                {2, 4, 1, 3},
                {5, 3, 6, 1},
                {6, 4, 5, 2},
                {6, 1, 5, 3},
        };

        Matrix B {
                {7, 5, 3, 2},
                {-1, 12, 40, -8},
                {9, -11, 2, 3},
                {4, 6, 2, 1}
        };

        double one_by_one;
        for (size_t i = 0; i < A.cols(); ++i) {
            one_by_one += loss.evaluate_1d(A(Eigen::all, i), B(Eigen::all, i));
        }

        one_by_one /= A.cols();

        assert(one_by_one == loss.evaluate_2d(A, B));
    }
}

void test_block() {
    {
        ComputeBlock test(2, 2);
        test.get_A();
        test.get_b();
        Vector v1 {{1, 1}};
        test.evaluate_1d(v1);
    }

    {
        Matrix A {
                {2, 4, 1, 3},
                {5, 3, 6, 1},
                {6, 4, 5, 2},
                {6, 1, 5, 3},
        };

        Matrix B {
                {7, 5, 3, 2},
                {-1, 12, 40, -8},
                {9, -11, 2, 3},
                {4, 6, 2, 1}
        };

        ComputeBlock test(4, 4);
        Matrix result_A = test.evaluate_2d(A);
        for (size_t i = 0; i < B.cols(); ++i) {
            assert(result_A(Eigen::all, i) == test.evaluate_1d(A(Eigen::all, i)));
        }

        Matrix result_B = test.evaluate_2d(B);
        for (size_t i = 0; i < B.cols(); ++i) {
            assert(result_B(Eigen::all, i) == test.evaluate_1d(B(Eigen::all, i)));
        }
    }
}

void test_all() {
    test_loss();
    test_block();
}