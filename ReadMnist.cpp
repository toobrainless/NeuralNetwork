#include "ReadMnist.h"

int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void ReadMNIST(int NumberOfImages, int DataOfAnImage, std::vector<std::vector<double>> &arr) {
    arr.resize(NumberOfImages, std::vector<double>(DataOfAnImage));
    std::ifstream file("../t10k-images-idx3-ubyte", std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for (int i = 0; i < number_of_images; ++i) {
            for (int r = 0; r < n_rows; ++r) {
                for (int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char *)&temp, sizeof(temp));
                    arr[i][(n_rows * r) + c] = (double)temp;
                }
            }
        }
    } else {
        std::cout << "Can't read MNIST\n";
    }
}

void ReadMNISTLABELS(std::vector<int> &labels) {
    std::ifstream file("../t10k-labels-idx1-ubyte", std::ios::binary);
    int magic_number = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    int number_of_labels;
    file.read((char *)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = ReverseInt(number_of_labels);
    labels.resize(number_of_labels);
    if (file.is_open()) {
        for (size_t i = 0; i < number_of_labels; ++i) {
            unsigned char temp = 0;
            file.read((char *)&temp, sizeof(temp));
            labels[i] = (double)temp;
        }
    }
}
