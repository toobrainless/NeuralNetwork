cmake_minimum_required(VERSION 3.20)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

include_directories(.)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(main main.cpp ComputeBlock.cpp Net.cpp)
target_link_libraries (main Eigen3::Eigen)
