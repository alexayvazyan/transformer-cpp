#include "transformer.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>

WeightStorage::WeightStorage(const ModelDimensions& d) {
    W_e.resize(d.d_words * d.d_embed);

    W_q.resize(d.d_embed * d.d_qkv);
    W_k.resize(d.d_embed * d.d_qkv);
    W_v.resize(d.d_embed * d.d_qkv);

    W_g.resize(d.d_neurons * d.d_embed);
    W_l.resize(d.d_neurons * d.d_embed);
    W_bg.resize(d.d_neurons);
    W_bl.resize(d.d_embed);

    W_u.resize(d.d_embed * d.d_words);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    auto init = [&](std::vector<float>& v) {
        for (float& x : v) x = dist(rng);
    };

    init(W_e); 
    init(W_q); 
    init(W_k); 
    init(W_v);
    init(W_g); 
    init(W_l); 
    init(W_u);

    std::fill(W_bg.begin(), W_bg.end(), 0.0f);
    std::fill(W_bl.begin(), W_bl.end(), 0.0f);
}