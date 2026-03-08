#include "helpers.h"
#include "helpers_eigen.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
using namespace Eigen;


auto mean(std::vector<float> vec) -> float {
    auto sum = 0.0f;
    for(unsigned i = 0; i<vec.size(); i++){
        sum+=vec.at(i);
    }
    return sum / vec.size();
}

auto sum_vec(std::vector<float> vec) -> float {
    auto sum = 0.0f;
    for(unsigned i = 0; i<vec.size(); i++){
        sum+=vec.at(i);
    }
    return sum;
}

auto pow(float base, int power) -> float {
    if(power == 0){
        return 1;
    }
    auto result = 1.0f;
    while(power>0){
        result*=base;
        power--;
    }
    return result;
}

auto calcrmseVecXf(VectorXf predictions, VectorXf target) -> float{
    auto n = static_cast<int>(predictions.size());
    auto sum = 0.0f;
    for(unsigned i = 0; i < n; i++){
        auto residual = predictions(i) - target(i);
        sum += residual * residual / n;
    }
    return sqrtf(sum);
}

auto testtrainsplit(MatrixXf A) -> std::vector<MatrixXf>{
    std::vector<MatrixXf> train_test;
    auto train = A.block(0,0,A.rows()/2,A.cols());
    auto test = A.block(A.rows()/2,0,A.rows()/2,A.cols());
    train_test.push_back(train);
    train_test.push_back(test);
    return train_test;
}

auto generatedummydata(int rows, int cols) -> MatrixXf{
    auto data = MatrixXf(rows, cols);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            data(i,j) = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    return data;
}

auto generatenotsodummydata(int rows, int cols) -> MatrixXf{
    auto dummydata = generatedummydata(rows, cols);
    for(int i = 0; i < rows; i++){
        dummydata(i,dummydata.cols()-1) = dummydata(i,0)*dummydata(i,0) - 5*dummydata(i,0) + 1.5*dummydata(i,1) * dummydata(i,2) * dummydata(i,3);
    }
    return dummydata;
}



auto merge_two_sorted_vectors(std::vector<float> a, std::vector<float> b) -> std::vector<float> {
    std::vector<float> fin_vec;
    size_t i = 0, j = 0;

    while (i < a.size() && j < b.size()) {
        if (a[i] <= b[j]) {
            fin_vec.push_back(a[i]);
            i+=1;
        } else {
            fin_vec.push_back(b[j]);
            j+=1;
        }
    }
    while (i < a.size()) {
        fin_vec.push_back(a[i]);
        i+=1;
    }

    while (j < b.size()) {
        fin_vec.push_back(b[j]);
        j+=1;
    }

    return fin_vec;
}

auto mergesort(std::vector<float> unsorted) -> std::vector<float> {
    auto middle = static_cast<int>(unsorted.size()/2);

    if (unsorted.size() < 2){
        return unsorted;
    }

    auto lefthalf = std::vector<float>(unsorted.begin(), unsorted.begin() + middle);
    auto righthalf = std::vector<float>(unsorted.begin() + middle, unsorted.end());

    auto sorted = merge_two_sorted_vectors(mergesort(lefthalf), mergesort(righthalf));
    return sorted;
} 



auto merge_two_sorted_vectors_index(std::vector<unsigned> a, std::vector<unsigned> b, std::vector<float> master) -> std::vector<unsigned> {
    std::vector<unsigned> fin_vec;
    size_t i = 0, j = 0;

    while (i < a.size() && j < b.size()) {
        if (master.at(a[i]) <= master.at(b[j])) {
            fin_vec.push_back(a[i]);
            i+=1;
        } else {
            fin_vec.push_back(b[j]);
            j+=1;
        }
    }
    while (i < a.size()) {
        fin_vec.push_back(a[i]);
        i+=1;
    }

    while (j < b.size()) {
        fin_vec.push_back(b[j]);
        j+=1;
    }

    return fin_vec;
}

auto mergesort_index(std::vector<unsigned> index_vector, std::vector<float> master) -> std::vector<unsigned> {
    auto middle = static_cast<int>(index_vector.size()/2);
    if (index_vector.size() < 2){
        return index_vector;
    }

    auto lefthalf = std::vector<unsigned>(index_vector.begin(), index_vector.begin() + middle);
    auto righthalf = std::vector<unsigned>(index_vector.begin() + middle, index_vector.end());

    auto sorted = merge_two_sorted_vectors_index(mergesort_index(lefthalf, master), mergesort_index(righthalf, master), master);
    return sorted;
} 

auto mergesort_index_master(std::vector<float> unsorted) -> std::vector<unsigned> {
    auto index_vector = std::vector<unsigned>{};
    for(unsigned i = 0; i < unsorted.size(); i++){
        index_vector.push_back(i);
    }
    return mergesort_index(index_vector, unsorted);
}

auto softmax(std::vector<float> v) -> std::vector<float>{
    auto result = std::vector<float>{};
    result.resize(v.size());
    float max_val = *std::max_element(v.begin(), v.end());

    float sum = 0.0f;
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = std::exp(v[i] - max_val);
        sum += result[i];
    }

    for (size_t i = 0; i < v.size(); ++i) {
        result[i] /= sum;
    }

    return result;
}