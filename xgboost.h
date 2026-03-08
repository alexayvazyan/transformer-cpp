#pragma once
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;

struct TreeInfo {
    std::vector<unsigned> e1; //first element is vector of indicies of root
    float e2 = -1.0f; //second element is similarity score
    bool e3 = false; //third element is bool of true if split or false if leaf
    std::vector<unsigned> e4; //fourth element is a vector of indicies in left split (given split)
    float e5 = 0.0f; //fifth element is the splitval or leafval if not split
    unsigned e6; // sixth element is the splitfeature
};

struct Tree {
    std::vector<std::vector<TreeInfo>> t1;
    Tree() : t1(5, std::vector<TreeInfo>(32)) {}
};

auto runtree(MatrixXf Data, Tree Tree) -> std::vector<float>;
auto xgboost(MatrixXf Data, float learning_rate = 0.3f, float l1_reg = 0.0f, float l2_reg = 0.0f, int maxdepth = 5, int maxtrees=50) -> std::tuple<float, float>;
auto buildxgboostmodel(MatrixXf Train, std::vector<Tree> Trees, float learning_rate, float l1_reg, float l2_reg, int maxdepth, int maxtrees) -> std::vector<Tree>;
auto testxgboostmodel(MatrixXf test, std::vector<Tree> Trees, float learning_rate) -> float;
auto buildtree(MatrixXf Train, float l1_reg, float l2_reg, int maxdepth) -> Tree;
auto buildtree_FAST(MatrixXf Train, float l1_reg, float l2_reg, int maxdepth) -> Tree;