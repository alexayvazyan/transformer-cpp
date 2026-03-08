#include "transformer.h"
#include <fstream>

void write_vec(std::ofstream& out, const std::vector<float>& v) {
    out.write(reinterpret_cast<const char*>(v.data()),
              v.size() * sizeof(float));
}

void save_weights(const WeightStorage& w, const char* path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return;

    write_vec(out, w.W_e);
    write_vec(out, w.W_q);
    write_vec(out, w.W_k);
    write_vec(out, w.W_v);
    write_vec(out, w.W_g);
    write_vec(out, w.W_l);
    write_vec(out, w.W_bg);
    write_vec(out, w.W_bl);
    write_vec(out, w.W_u);
}
