#include "matrixlin.h"
#include "xgboost.h"
#include "helpers.h"
#include "helpers_eigen.h"
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <chrono> 
using namespace Eigen;


int main(){
    auto start = std::chrono::high_resolution_clock::now();
    auto data = generatenotsodummydata(1000,100); //last column is target
    auto [xgtrain_rmse, xgtest_rmse] = xgboost(data, 0.3f, 0.01f, 0.01f, 5, 200); //data, learining_rate, l1_reg, l2_reg, max_depth, max_trees
    auto [coeffs, mtrain_rmse, mtest_rmse] = matrixlinreg(data);
    std::cout << "xgboost train rmse:" << xgtrain_rmse << "\n";
    std::cout << "xgboost test rmse:" << xgtest_rmse << "\n";
    std::cout << "least squares linear train rmse:" << mtrain_rmse << "\n";
    std::cout << "least squares linear test rmse:" << mtest_rmse << "\n";
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - start;
    std::cout << "Took " << runtime.count() << " seconds" << "\n";
    return 0;
}

// int main(){
//     auto data = generatedummydata(100,5); //last column is target
//     auto [coeffs, train_rmse, test_rmse] = matrixlinreg(data); 
//     std::cout << coeffs << "\n";
//     std::cout << train_rmse << "\n";
//     std::cout << test_rmse << "\n";
//     return 0;
// }