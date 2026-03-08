#include "matrixlin.h"
#include "helpers.h"
#include "helpers_eigen.h"
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>
using namespace Eigen;

auto matrixlinreg(MatrixXf Data) -> std::tuple<VectorXf, float, float>{
    Data.conservativeResize(Data.rows(), Data.cols()+1); 
    auto intercept_vector = VectorXf::Ones(Data.rows());
    Data.rightCols(Data.cols() - 1) = Data.leftCols(Data.cols() - 1).eval();
    Data.col(0) = intercept_vector;
    auto traintest = testtrainsplit(Data);
    auto train = traintest.at(0);
    auto test = traintest.at(1);

    auto train_targets = train.col(train.cols() - 1);
    auto train_features = train.block(0,0,train.rows(),train.cols() - 1);
    auto test_targets = test.col(test.cols() - 1);
    auto test_features = test.block(0,0,test.rows(),test.cols() - 1);
    VectorXf coeffs = (train_features.transpose() * train_features).inverse() * train_features.transpose() * train_targets;
    auto train_rmse = calcrmseVecXf(train_features*coeffs, train_targets);
    auto test_rmse = calcrmseVecXf(test_features*coeffs, test_targets);
    return std::make_tuple(coeffs, train_rmse, test_rmse);
}   


