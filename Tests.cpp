#include "Tests.h"

namespace {
using Vector = Net::Vector;
using Matrix = Net::Matrix;
}

void test_loss() {
    std::cout << "test_loss\n";
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

    std::cout << "passed test_loss\n";
}

void test_block() {
    std::cout << "test_block\n";
    {
        ComputeBlock test(2, 2, "sigmoid");
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

        ComputeBlock test(4, 4, "sigmoid");
        Matrix result_A = test.evaluate_2d(A);
        for (size_t i = 0; i < B.cols(); ++i) {
            assert(result_A(Eigen::all, i) == test.evaluate_1d(A(Eigen::all, i)));
        }

        Matrix result_B = test.evaluate_2d(B);
        for (size_t i = 0; i < B.cols(); ++i) {
            assert(result_B(Eigen::all, i) == test.evaluate_1d(B(Eigen::all, i)));
        }
    }

    std::cout << "passed test_block\n";
}

void test_sigmoid() {
    std::cout << "test_sigmoid\n";
    {
        Net test({3, 4, 2}, {"sigmoid", "sigmoid"});
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

    std::cout << "passed test_sigmoid\n";
}

void test_relu() {
    std::cout << "test_relu\n";

    {
        Net test({5, 4, 3, 2, 1}, {"relu", "relu", "relu", "relu"}, 1e-6, 1e-5);
        Matrix m1 {
                {3, 15, 45, 25, 30},
                {20, 1, 25, 6, 12},
                {1, 4, 7, 7, 8},
                {2, 8, 9, 8, 17},
                {25, 6, 7, 9, 11}
        };

        Matrix v1 {
                {3 + 20 + 1 + 2 + 25, 15 + 1 + 4 + 8 + 6, 45 + 25 + 7 + 9 + 7, 25 + 6 + 7 + 8 + 9, 30 + 12 + 8 + 17 + 11},
        };

        Vector v2 {{1, 2, 3, 4, 5}};
        Vector v3 {{2, 6, 8, 14, 25}};
        Vector v4 {{1, 1, 1, 1, 1}};
        Vector v5 {{2, 10, 14, 17, 9}};

        test.train(m1, v1);

        std::cout << test.predict_1d(v2) << '\n';
        std::cout << test.predict_1d(v3) << '\n';
        std::cout << test.predict_1d(v4) << '\n';
        std::cout << test.predict_1d(v5) << '\n';
        std::cout << "\n\n";
    }

    std::cout << "passed test_relu\n";
}

void test_mnist() {
    std::vector<std::vector<double>> images;
    std::vector<int> labels;
    ReadMNIST(10000, 784, images);
    ReadMNISTLABELS(labels);

//    8500 -- train | 1500 -- test
//firstly 100 test
    Matrix m1(784, 100); // images
    for (size_t i = 0; i < 100; ++i) {
        for (size_t j = 0; j < 784; ++j) {
//            if (images[i][j] == 0) {
//                m1(j, i) = 0;
//            } else {
//                m1(j, i) = 1;
//            }
            m1(j, i) = images[i][j] / 255;
        }
    }

    Matrix answers = Matrix::Zero(10, 100);
    for (size_t i = 0; i < 100; ++i) {
        answers(labels[i], i) = 1;
    }

//    std::cout << answers << '\n';

    Net test({784, 100, 50, 10}, "relu", 1e-1, 0.01);

    test.train(m1, answers);

    std::cout << "done!\n";

    std::vector<Vector> test_train(100, Vector::Zero(784));

    for (size_t i = 100; i < 200; ++i) {
        for (size_t j = 0; j < 784; ++j) {
//            if (images[i][j] == 0) {
//                test_train[i - 100](j) = 0;
//            } else {
//                test_train[i - 100](j) = 1;
//            }
            test_train[i - 100](j) = images[i][j] / 255;
        }
    }

    for (size_t i = 0; i < 100; ++i) {
        Vector ans = test.predict_1d(test_train[i]);
        std::cout << "------------------------\n";
        std::cout << "------------------------\n";
        std::cout << ans << '\n';
        std::cout << "------------------------\n";
        std::cout << labels[i + 100] << '\n';
        std::cout << "------------------------\n";
        std::cout << "------------------------\n";
    }

}

void test_all() {
    test_loss();
    test_block();
    test_sigmoid();
    test_relu();
//    test_mnist();
}
