#include "Tests.h"

namespace {
using Vector = Net::Vector;
using Matrix = Net::Matrix;
}  // namespace

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
        Matrix A{
            {2, 4, 1, 3},
            {5, 3, 6, 1},
            {6, 4, 5, 2},
            {6, 1, 5, 3},
        };

        Matrix B{{7, 5, 3, 2}, {-1, 12, 40, -8}, {9, -11, 2, 3}, {4, 6, 2, 1}};

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
        Vector v1{{1, 1}};
        test.evaluate_1d(v1);
    }

    {
        Matrix A{
            {2, 4, 1, 3},
            {5, 3, 6, 1},
            {6, 4, 5, 2},
            {6, 1, 5, 3},
        };

        Matrix B{{7, 5, 3, 2}, {-1, 12, 40, -8}, {9, -11, 2, 3}, {4, 6, 2, 1}};

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
        Matrix m1{{3, 15, 45}, {20, 1, 25}, {1, 4, 7}};

        Matrix v1{{10 / 120, 2 / 12, 4 / 123}, {3 / 9, 4 / 7, 12 / 56}};

        test.train(m1, v1);
    }

    std::cout << "passed test_sigmoid\n";
}

void test_relu() {
    std::cout << "test_relu\n";

    {
        Net test({5, 1}, {"relu"}, 1e-6, 1e-4);
        Matrix m1{{3, 15, 45, 25, 30},
                  {20, 1, 25, 6, 12},
                  {1, 4, 7, 7, 8},
                  {2, 8, 9, 8, 17},
                  {25, 6, 7, 9, 11}};

        Matrix v1{
            {3 + 20 + 1 + 2 + 25, 15 + 1 + 4 + 8 + 6, 45 + 25 + 7 + 9 + 7, 25 + 6 + 7 + 8 + 9,
             30 + 12 + 8 + 17 + 11},
        };

        Vector v2{{1, 2, 3, 4, 5}};
        Vector v3{{2, 6, 8, 14, 25}};
        Vector v4{{1, 1, 1, 1, 1}};
        Vector v5{{2, 10, 14, 17, 9}};

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
    std::cout << "test_mnist\n";

    std::vector<std::vector<double>> images;
    std::vector<int> labels;
    ReadMNIST(10000, 784, images);
    ReadMNISTLABELS(labels);

    std::cout << "point1\n";

    //    8500 -- train | 1500 -- test
    // firstly 100 test
    Matrix m1(784, 8500);  // images
    for (size_t i = 0; i < 8500; ++i) {
        for (size_t j = 0; j < 784; ++j) {
            m1(j, i) = images[i][j] / 255;
        }
    }

    std::cout << "point2\n";

    Matrix answers = Matrix::Zero(10, 8500);
    for (size_t i = 0; i < 8500; ++i) {
        answers(labels[i], i) = 1;
    }

    std::cout << "point3\n";

    Net test({784, 16, 16, 10}, {"relu", "relu", "softmax"}, 1e-1, 1e-4);

    std::cout << "point4\n";

    test.train(m1, answers);

    std::cout << "done!\n";

    std::vector<Vector> test_train(1500, Vector::Zero(784));

    for (size_t i = 8500; i < 10000; ++i) {
        for (size_t j = 0; j < 784; ++j) {
            test_train[i - 8500](j) = images[i][j] / 255;
        }
    }

    size_t hit_rate = 0;

    for (size_t i = 0; i < 1500; ++i) {
        Vector ans = test.predict_1d(test_train[i]);
        double cur_max = 0;
        size_t max_index = 0;
        for (size_t i = 0; i < 10; ++i) {
            if (ans(i) > cur_max) {
                cur_max = ans(i);
                max_index = i;
            }
        }

        if (max_index == labels[i + 8500]) {
            hit_rate += 1;
        }
    }

    std::cout << hit_rate << '\n';
}

void test_all() {
    //    test_loss();
    //    test_block();
    //    test_sigmoid();
    //    test_relu();
    test_mnist();
}
