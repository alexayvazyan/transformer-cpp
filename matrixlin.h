#pragma once
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;

auto matrixlinreg(MatrixXf A) -> std::tuple<VectorXf, float, float>;
