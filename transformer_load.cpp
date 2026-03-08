#include <fstream>
#include <stdexcept>
#include "transformer.h"

static void read_vec(std::ifstream& in, std::vector<float>& v) {
    in.read(reinterpret_cast<char*>(v.data()),
            v.size() * sizeof(float));

    if (!in) {
        throw std::runtime_error("Failed to read weights");
    }
}

WeightStorage load_weights(const char* path, const ModelDimensions& d) {
    WeightStorage w(d);  // allocates correct sizes

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open weights.bin");
    }

    read_vec(in, w.W_e);
    read_vec(in, w.W_q);
    read_vec(in, w.W_k);
    read_vec(in, w.W_v);
    read_vec(in, w.W_g);
    read_vec(in, w.W_l);
    read_vec(in, w.W_bg);
    read_vec(in, w.W_bl);
    read_vec(in, w.W_u);

    return w;
}
