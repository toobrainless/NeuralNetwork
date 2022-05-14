#include "Tests.h"

namespace {
using Vector = Net::Vector;
using Matrix = Net::Matrix;
}

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
        ActivationFunction* activation_function = new Sigmoid;
        ComputeBlock test(2, 2, activation_function);
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

        ActivationFunction* activation_function = new Sigmoid;
        ComputeBlock test(4, 4, activation_function);
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

void test_sigmoid() {
    {
        Net test({3, 4, 2}, "sigmoid");
        Matrix m1 {
                {3, 15, 45},
                {20, 1, 25},
                {1, 4, 7}
        };

        Matrix v1 {
                {10/120, 2/12, 4/123},
                {3/9, 4/7, 12/56}
        };

        test.train(m1, v1);
    }
}

void test_relu() {
    {
        Net test({5, 1}, "relu", 1e-6, 1e-4);
        Matrix m1 {
                {3, 15, 45, 25},
                {20, 1, 25, 6},
                {1, 4, 7, 7},
                {2, 8, 9, 8},
                {25, 6, 7, 9}
        };

        Matrix v1 {
                {3 + 20 + 1 + 2 + 25, 15 + 1 + 4 + 8 + 6, 45 + 25 + 7 + 9 + 7, 25 + 6 + 7 + 7 + 9},
        };

        Vector v2 {{1, 2, 3, 4, 5}};
        Vector v3 {{2, 6, 8, 14, 25}};
        Vector v4 {{1, 1, 1, 1, 1}};

        test.train(m1, v1);

        std::cout << test.predict_1d(v2) << '\n';
        std::cout << test.predict_1d(v3) << '\n';
        std::cout << test.predict_1d(v4) << '\n';
    }
}

void test_all() {
    test_loss();
    test_block();
//    test_sigmoid();
    test_relu();
}
