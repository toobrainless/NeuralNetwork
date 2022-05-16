#pragma once
#include <iostream>
#include <vector>
#include <fstream>

int ReverseInt(int i);
void ReadMNIST(int NumberOfImages, int DataOfAnImage, std::vector<std::vector<double>>& arr);
void ReadMNISTLABELS(std::vector<int>& labels);