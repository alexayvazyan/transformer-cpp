#include "test.h"
#include <vector>
#include <iostream>
int main(){
    auto vec1 = std::vector<float>{1,2,3,4};
    auto vec2 = std::vector<float>{3,5,7,9};
    auto soln = linreguni(vec1,vec2);
    std::cout << soln.at(0) << soln.at(1) << "\n";
    return 0;
}