#include "transformer.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>

int main() {
    ModelDimensions dims;
    dims.d_words = 10000;
    dims.d_embed = 128;
    dims.d_qkv = 128;
    dims.d_neurons = 256;
    WeightStorage weights(dims);
    std::cout << "total weights size: " << weights.W_e.size()
    + weights.W_q.size()
    + weights.W_k.size()
    + weights.W_v.size()
    + weights.W_g.size()
    + weights.W_l.size()
    + weights.W_bg.size()
    + weights.W_bl.size()
    + weights.W_u.size() << "\n";
    save_weights(weights, "/Users/alexanderayvazyan/Documents/cpplearning/project/weights.bin");
    return 0;
}