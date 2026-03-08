#include <iostream>
#include "test.h"
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;

auto sumvector(std::vector<float> vec1) -> float{
    auto sum = 0;
    for(auto j : vec1){
        sum += j;
    }
    return sum;
}

auto sumvectorsq(std::vector<float> vec1) -> float{
    auto sum = 0;
    for(auto j : vec1){
        sum += j*j;
    }
    return sum;
}

auto linreguni(std::vector<float> x, std::vector<float> y) -> std::vector<float>{
    auto n = static_cast<int>(y.size());
    auto ysum = sumvector(y);
    auto ymean = static_cast<float>(ysum/n);
    auto xsum = sumvector(x);
    auto xmean = static_cast<float>(xsum/n);
    auto sxx = static_cast<float>(0);
    auto sxy = static_cast<float>(0);

    for (auto j = 0; j < n; j ++){
        sxx += (x.at(j) - xmean) * (x.at(j)-xmean);
        sxy += (y.at(j) - ymean) * (x.at(j)-xmean);
    }
    auto coeffs = std::vector<float> {sxy/sxx, ymean - xmean * sxy/sxx};
    return coeffs;
}
