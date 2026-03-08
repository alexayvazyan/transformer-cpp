#pragma once
#include <vector>
#include <Eigen/Dense>

auto testtrainsplit(Eigen::MatrixXf A) -> std::vector<Eigen::MatrixXf>;
auto generatedummydata(int rows, int cols) -> Eigen::MatrixXf;
auto calcrmseVecXf(Eigen::VectorXf predictions, Eigen::VectorXf target) -> float;
auto generatenotsodummydata(int rows, int cols) -> Eigen::MatrixXf;
