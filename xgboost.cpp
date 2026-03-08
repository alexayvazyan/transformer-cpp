#include "xgboost.h"
#include "helpers.h"
#include "helpers_eigen.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
using namespace Eigen;

auto xgboost(MatrixXf Data, float learning_rate, float l1_reg, float l2_reg, int maxdepth, int maxtrees) -> std::tuple<float, float>{
    auto traintest = testtrainsplit(Data);
    auto train = traintest.at(0);
    auto test = traintest.at(1);
    auto Trees = std::vector<Tree>{};
    // std::cout << train.col(train.cols()-1) << "\n";
    auto traintrees = buildxgboostmodel(train, Trees, learning_rate, l1_reg, l2_reg, maxdepth, maxtrees);
    auto trainrmse = testxgboostmodel(train, traintrees, learning_rate);
    auto testrmse = testxgboostmodel(test, traintrees, learning_rate);
    return std::tuple{trainrmse, testrmse};
}

auto buildxgboostmodel(MatrixXf Train, std::vector<Tree> Trees, float learning_rate, float l1_reg, float l2_reg, int maxdepth, int maxtrees) -> std::vector<Tree>{
    auto new_tree = buildtree_FAST(Train, l1_reg, l2_reg, maxdepth);
    Trees.push_back(new_tree);
    auto predictions = runtree(Train, new_tree);
    // for(unsigned i = 0; i < predictions.size(); i++){
    //     std::cout<<predictions.at(i)<<"\n";
    // }
    VectorXf residuals(predictions.size());
    for(unsigned i = 0; i < predictions.size(); i++){
        residuals(i) = Train(i, Train.cols()-1) - learning_rate * predictions.at(i);
    }
    Train.col(Train.cols()-1) = residuals;
    if(Trees.size() > maxtrees){
        return Trees;
    }
    std::cout << "residuals.norm()  " << residuals.norm() << "\n";
    // std::cout << residuals << "\n";
    return buildxgboostmodel(Train, Trees, learning_rate, l1_reg, l2_reg, maxdepth, maxtrees);
}

auto testxgboostmodel(MatrixXf test, std::vector<Tree> Trees, float learning_rate) -> float{
    VectorXf final_predictions(test.rows());
    auto last_tree_preds = runtree(test, Trees.at(static_cast<int>(Trees.size()) - 1));
    for(unsigned i = 0; i < test.rows(); i++){
        final_predictions(i) = last_tree_preds.at(i);
    }
    for(auto i = static_cast<int>(Trees.size()) - 2; i>=0; i--){
        auto predictions = runtree(test, Trees.at(i));
        for(unsigned j = 0; j<predictions.size(); j++){
            final_predictions(j) = final_predictions(j) + learning_rate*predictions.at(j);
        }
    }
    auto targets = test.col(test.cols()-1);
    // for(unsigned i = 0; i < final_predictions.size(); i++){
    //     std::cout<<final_predictions(i)<<"\n";
    // }
    return calcrmseVecXf(final_predictions, targets);
}

auto runtree(MatrixXf Data, Tree Tree) -> std::vector<float>{
    auto predictions = std::vector<float> {};
    for(unsigned i = 0; i < Data.rows(); i++){
        auto k=0; //row index
        for(unsigned j = 0; j < Tree.t1.size(); j++){
            // std::cout << "i" << i << " j" << j << " k" << k << "\n";
            if(Tree.t1.at(j).at(k).e3 == false){
                predictions.push_back(Tree.t1.at(j).at(k).e5);
                // std::cout << "added " << Tree.t1.at(j).at(k).e5 << " to predictions" << "\n";
                break;
            }
            // std::cout << "Compared " << Data(i,Tree.t1.at(j).at(k).e6) << " to " << Tree.t1.at(j).at(k).e5 << " on feature " << Tree.t1.at(j).at(k).e6 << "\n";
            if(Data(i,Tree.t1.at(j).at(k).e6) < Tree.t1.at(j).at(k).e5){
                k=2*k;
            } else {
                k=2*k+1;
            }
        }
    }
    return predictions;
}

auto buildtree(MatrixXf Train, float l1_reg, float l2_reg, int maxdepth) -> Tree{
    Tree treege;
    auto scores = std::vector<float> (Train.col(Train.cols()-1).data(), Train.col(Train.cols()-1).data() + Train.rows());
    auto starting_indicies = std::vector<unsigned> {};
    auto sum = 0.0f;
    auto max_depth = maxdepth;
    auto max_leafs = pow(2, max_depth);
    
    for(unsigned i = 0; i<scores.size(); i++){
        starting_indicies.push_back(i);
        sum += scores.at(i);
        // std::cout << scores.at(i) << "\n";
    }
    treege.t1.at(0).at(0).e1 = starting_indicies;
    treege.t1.at(0).at(0).e2 = sum * sum / (scores.size() + l2_reg);
    for(auto depth = 0; depth<max_depth; depth++){
        for(auto row = 0; row<max_leafs; row++){
            auto obs = treege.t1.at(depth).at(row).e1;
            auto similarity_score_root = treege.t1.at(depth).at(row).e2;

            if (similarity_score_root < 0){
                continue; // skip this node if there is no split leading to this node
            }
            auto numobs = static_cast<int>(obs.size());

            auto subset_matrix = Eigen::MatrixXf{numobs, Train.cols()};
            for(unsigned i = 0; i<numobs; i++){
                subset_matrix.row(i) = Train.row(obs.at(i));
            }
            auto subset_matrix_scores = std::vector<float> (subset_matrix.col(subset_matrix.cols()-1).data(), subset_matrix.col(subset_matrix.cols()-1).data() + subset_matrix.rows());
            
            if (numobs == 1){
                treege.t1.at(depth).at(row).e3 = false;
                treege.t1.at(depth).at(row).e5 = mean(subset_matrix_scores);
                continue; // force leaf if 1 observation
            }

            auto best_gain = -1000000000.0f;
            auto best_leftsplit_indicies = std::vector<unsigned>{};
            auto best_rightsplit_indicies = std::vector<unsigned>{};
            auto best_similarity_score_left = 0.0f;
            auto best_similarity_score_right = 0.0f;
            auto best_splitval = 0.0f;
            auto best_feature = 0u;

            for(unsigned i = 0; i<subset_matrix.cols()-1; i++){ //iterates over all features

                auto currfeature = std::vector<float> (subset_matrix.col(i).data(), subset_matrix.col(i).data() + subset_matrix.rows());
                auto sorted_currfeature = mergesort(currfeature);


                auto start = std::chrono::high_resolution_clock::now();

                for(unsigned j = 0; j<currfeature.size()-1; j++){ //iterates over all splits
                    auto splitval =(sorted_currfeature.at(j) + sorted_currfeature.at(j+1))/2;
                    auto leftsplit_indicies = std::vector<unsigned>{};
                    auto rightsplit_indicies = std::vector<unsigned>{};
                    auto sum_scores_left = 0.0f;
                    auto sum_scores_right = 0.0f;
                    for(int k = 0; k < static_cast<int>(subset_matrix.rows()); k++){ //iterates over data to compute similarity scores given the split
                        if(currfeature.at(k) < splitval){ 
                            sum_scores_left += subset_matrix_scores.at(k);
                            leftsplit_indicies.push_back(obs.at(k));
                        } else {
                            sum_scores_right += subset_matrix_scores.at(k);
                            rightsplit_indicies.push_back(obs.at(k));
                        }
                    }
                    auto similarity_score_left = sum_scores_left * sum_scores_left / (static_cast<int>(leftsplit_indicies.size()) + l2_reg);
                    auto similarity_score_right = sum_scores_right * sum_scores_right / (static_cast<int>(rightsplit_indicies.size()) + l2_reg);
                    auto gain = similarity_score_left + similarity_score_right - similarity_score_root - l1_reg;
                    // std::cout << "gain" << gain << "\n";
                    // if(gain < 0){
                    //     for(unsigned a = 0; a < leftsplit_indicies.size(); a++){
                    //         std::cout << subset_matrix_scores.at(leftsplit_indicies.at(a)) << "\n";
                    //     }
                    //     for(unsigned a = 0; a < rightsplit_indicies.size(); a++){
                    //         std::cout << subset_matrix_scores.at(rightsplit_indicies.at(a)) << "\n";
                    //     }
                    //     std::cout << similarity_score_left << sum_scores_left << static_cast<int>(leftsplit_indicies.size()) << "\n";
                    //     std::cout << similarity_score_right << sum_scores_right << static_cast<int>(rightsplit_indicies.size()) << "\n";
                    //     std::cout << "ssroot" << similarity_score_root << "\n";
                    //     for(unsigned a = 0; a < subset_matrix_scores.size(); a++){
                    //         std::cout << subset_matrix_scores.at(a) << "\n";
                    //     }
                    //     std::cout << "dog" << "\n";
                    // }
                    if(gain > best_gain){
                        best_gain = gain;
                        best_leftsplit_indicies = leftsplit_indicies;
                        best_rightsplit_indicies = rightsplit_indicies;
                        best_similarity_score_left = similarity_score_left;
                        best_similarity_score_right = similarity_score_right;
                        best_splitval = splitval;
                        best_feature = i;
                    }
                auto end = std::chrono::high_resolution_clock::now();
                }
            }
            // if(best_gain < 0){
            //     std::cout << "bestgain" << best_gain << "\n";
            //     std::cout << "depth" << depth << "\n";
            //     std::cout << "row" << row << "\n";
            // }
            if(best_gain>0 and depth < max_depth-1){
                treege.t1.at(depth).at(row).e3 = true;
                treege.t1.at(depth).at(row).e4 = best_leftsplit_indicies;
                treege.t1.at(depth).at(row).e5 = best_splitval;
                treege.t1.at(depth).at(row).e6 = best_feature;
                treege.t1.at(depth+1).at(2*row).e1 = best_leftsplit_indicies;
                treege.t1.at(depth+1).at(2*row).e2 = best_similarity_score_left;
                treege.t1.at(depth+1).at(2*row+1).e1 = best_rightsplit_indicies;
                treege.t1.at(depth+1).at(2*row+1).e2 = best_similarity_score_right;
            } else {
                treege.t1.at(depth).at(row).e3 = false;
                treege.t1.at(depth).at(row).e5 = mean(subset_matrix_scores);
            }
        }
    }

    // std::cout << "printing tree" << "\n";
    // for(auto depth = 0; depth<max_depth; depth++){
    //     for(auto row = 0; row< pow(2, depth); row++){
    //         std::cout << "depth" << depth << "row" << row << "\n";
    //         std::cout << "ss" << treege.t1.at(depth).at(row).e2 << "\n";
    //         std::cout << "split" << treege.t1.at(depth).at(row).e3 << "\n";
    //         std::cout << "hessians" << treege.t1.at(depth).at(row).e1.size() << "\n";
    //         std::cout << "splitval/leafval" << treege.t1.at(depth).at(row).e5 << "\n";
    //         for(unsigned i = 0; i < treege.t1.at(depth).at(row).e1.size(); i++){
    //             std::cout << "value" << treege.t1.at(depth).at(row).e1.at(i) << "\n";
    //         }
    //     }
    // }
    return treege;
}


auto buildtree_FAST(MatrixXf Train, float l1_reg, float l2_reg, int maxdepth) -> Tree{
    Tree treege;
    auto scores = std::vector<float> (Train.col(Train.cols()-1).data(), Train.col(Train.cols()-1).data() + Train.rows());
    auto starting_indicies = std::vector<unsigned> {};
    starting_indicies.reserve(scores.size());
    auto sum = 0.0f;
    auto max_depth = maxdepth;
    auto max_leafs = pow(2, max_depth);
    
    for(unsigned i = 0; i<scores.size(); i++){
        starting_indicies.push_back(i);
        sum += scores.at(i);
        // std::cout << scores.at(i) << "\n";
    }
    treege.t1.at(0).at(0).e1 = starting_indicies;
    treege.t1.at(0).at(0).e2 = sum * sum / (scores.size() + l2_reg);
    for(auto depth = 0; depth<max_depth; depth++){
        for(auto row = 0; row< pow(2, depth+1); row++){
            auto obs = treege.t1.at(depth).at(row).e1;
            auto similarity_score_root = treege.t1.at(depth).at(row).e2;

            if (similarity_score_root < 0){
                continue; // skip this node if there is no split leading to this node
            }
            auto numobs = static_cast<int>(obs.size());

            auto subset_matrix = Eigen::MatrixXf{numobs, Train.cols()};
            for(unsigned i = 0; i<numobs; i++){
                subset_matrix.row(i) = Train.row(obs.at(i));
            }
            auto subset_matrix_scores = std::vector<float> (subset_matrix.col(subset_matrix.cols()-1).data(), subset_matrix.col(subset_matrix.cols()-1).data() + subset_matrix.rows());
            
            if (numobs == 1){
                treege.t1.at(depth).at(row).e3 = false;
                treege.t1.at(depth).at(row).e5 = mean(subset_matrix_scores);
                continue; // force leaf if 1 observation
            }

            auto best_similarity_score_left = 0.0f;
            auto best_similarity_score_right = 0.0f;
            auto best_i = 0u;
            auto best_j = 0u;
            auto sorted_feature_indicies = std::vector<std::vector<unsigned>>{};
            auto top_score = -1e9f;
            for(unsigned i = 0; i < subset_matrix.cols()-1; i++){ //iterates over all features
                auto currfeature = std::vector<float> (subset_matrix.col(i).data(), subset_matrix.col(i).data() + subset_matrix.rows());
                sorted_feature_indicies.push_back(mergesort_index_master(currfeature));
            }
            
            for(unsigned i = 0; i < subset_matrix.cols()-1; i++){
                auto hessians_left = 1;
                auto hessians_right = subset_matrix.rows() - 1;
                auto feature = sorted_feature_indicies.at(i);
                auto grad_left = subset_matrix_scores.at(feature.at(0));
                auto grad_right = sum_vec(subset_matrix_scores) - grad_left;
                top_score = (grad_right*grad_right)/(hessians_right + l2_reg) + (grad_left)*(grad_left)/(hessians_left + l2_reg);
                for(unsigned j = 1; j < subset_matrix.rows()-1; j++){
                    hessians_left += 1;
                    hessians_right -= 1;
                    auto new_inclusion = subset_matrix_scores.at(feature.at(j));
                    grad_left += new_inclusion;
                    grad_right -= new_inclusion;
                    auto sim_score_left = (grad_left*grad_left)/(hessians_left + l2_reg);
                    auto sim_score_right = (grad_right*grad_right)/(hessians_right + l2_reg);
                    auto score = sim_score_left + sim_score_right;
                    if(score > top_score){
                        top_score = score;
                        best_similarity_score_left = sim_score_left;
                        best_similarity_score_right = sim_score_right;
                        best_i = i;
                        best_j = j;
                    }
                }
            }
            // std::cout << best_j << "bestbestbest" << "\n";
            auto best_feature_indicies = sorted_feature_indicies.at(best_i);
            auto best_splitval = (subset_matrix(best_feature_indicies.at(best_j), best_i) + subset_matrix(best_feature_indicies.at(best_j+1), best_i))/2;
            auto best_leftsplit_indicies = std::vector<unsigned> (best_feature_indicies.begin(), best_feature_indicies.begin() + best_j + 1);
            auto best_rightsplit_indicies = std::vector<unsigned> (best_feature_indicies.begin() + best_j + 1, best_feature_indicies.end());

            auto best_leftsplit_indicies_wrt_train = std::vector<unsigned> {};
            auto best_rightsplit_indicies_wrt_train = std::vector<unsigned> {};

            for(unsigned z = 0; z<best_leftsplit_indicies.size(); z++){
                best_leftsplit_indicies_wrt_train.push_back(obs.at(best_leftsplit_indicies.at(z)));
            }
            for(unsigned z = 0; z<best_rightsplit_indicies.size(); z++){
                best_rightsplit_indicies_wrt_train.push_back(obs.at(best_rightsplit_indicies.at(z)));
            }

            auto best_gain = top_score - similarity_score_root - l1_reg;
            if(best_gain>0 and depth < max_depth-1){
                // for(unsigned dog = 0; dog < best_leftsplit_indicies.size(); dog++){
                //     std::cout << "best left indicies " << best_leftsplit_indicies.at(dog) << "\n";
                //     std::cout << subset_matrix_scores.at(best_feature_indicies.at(best_leftsplit_indicies.at(dog))) << "\n";
                // }
                // for(unsigned dog = 0; dog < best_rightsplit_indicies.size(); dog++){
                //     std::cout << "best right indicies " << best_rightsplit_indicies.at(dog) << "\n";
                //     std::cout << subset_matrix_scores.at(best_feature_indicies.at(best_rightsplit_indicies.at(dog))) << "\n";
                // }
                // std::cout << "depth " << depth << "row " << row << "\n";
                treege.t1.at(depth).at(row).e3 = true;
                treege.t1.at(depth).at(row).e4 = best_leftsplit_indicies_wrt_train;
                treege.t1.at(depth).at(row).e5 = best_splitval;
                treege.t1.at(depth).at(row).e6 = best_i;
                treege.t1.at(depth+1).at(2*row).e1 = best_leftsplit_indicies_wrt_train;
                treege.t1.at(depth+1).at(2*row).e2 = best_similarity_score_left;
                treege.t1.at(depth+1).at(2*row+1).e1 = best_rightsplit_indicies_wrt_train;
                treege.t1.at(depth+1).at(2*row+1).e2 = best_similarity_score_right;
            } else {
                treege.t1.at(depth).at(row).e3 = false;
                treege.t1.at(depth).at(row).e5 = mean(subset_matrix_scores);
            }
        }
    }
    // std::cout << "printing tree" << "\n";
    // for(auto depth = 0; depth<max_depth; depth++){
    //     for(auto row = 0; row< pow(2, depth); row++){
    //         std::cout << "depth " << depth << "row " << row << "\n";
    //         std::cout << "ss " << treege.t1.at(depth).at(row).e2 << "\n";
    //         std::cout << "split " << treege.t1.at(depth).at(row).e3 << "\n";
    //         std::cout << "hessians " << treege.t1.at(depth).at(row).e1.size() << "\n";
    //         std::cout << "splitval/leafval " << treege.t1.at(depth).at(row).e5 << "\n";
    //         for(unsigned i = 0; i < treege.t1.at(depth).at(row).e1.size(); i++){
    //             std::cout << "value " << treege.t1.at(depth).at(row).e1.at(i) << "\n";
    //         }
    //         if(treege.t1.at(depth).at(row).e3){
    //             std::cout << "feautre " << treege.t1.at(depth).at(row).e6 << "\n";
    //         }
    //     }
    // }
    return treege;
}
