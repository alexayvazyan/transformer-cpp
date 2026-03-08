#pragma once
#include <vector>

auto merge_two_sorted_vectors(std::vector<float>, std::vector<float>) -> std::vector<float>;
auto mergesort(std::vector<float>) -> std::vector<float>;
auto mean(std::vector<float> vec) -> float;
auto pow(float base, int power) -> float;
auto mergesort_index(std::vector<unsigned> index_vector, std::vector<float> master) -> std::vector<unsigned>;
auto merge_two_sorted_vectors_index(std::vector<unsigned> a, std::vector<unsigned> b, std::vector<float> master) -> std::vector<unsigned>;
auto mergesort_index_master(std::vector<float> unsorted) -> std::vector<unsigned>;
auto sum_vec(std::vector<float> vec) -> float;
auto softmax(std::vector<float> v) -> std::vector<float>;