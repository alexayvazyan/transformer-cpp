#pragma once
#include <vector>

struct ModelDimensions {
    int d_words;   // vocab size
    int d_embed;   // embedding dim (must equal d_qkv for residual connection)
    int d_neurons; // MLP hidden dim
    int d_qkv;     // Q/K/V projection dim
};

struct WeightStorage {
    std::vector<float> W_e;  // embedding:          (d_words  × d_embed)
    std::vector<float> W_q;  // query projection:   (d_qkv    × d_embed)
    std::vector<float> W_k;  // key projection:     (d_qkv    × d_embed)
    std::vector<float> W_v;  // value projection:   (d_qkv    × d_embed)
    std::vector<float> W_g;  // MLP global:         (d_neurons × d_embed)
    std::vector<float> W_l;  // MLP local:          (d_neurons × d_embed)
    std::vector<float> W_bg; // MLP global bias:    (d_neurons)
    std::vector<float> W_bl; // MLP local bias:     (d_embed)
    std::vector<float> W_u;  // unembedding:        (d_embed  × d_words)
    WeightStorage(const ModelDimensions& d);
};

// Activations saved during forward pass, needed for backprop.
struct ForwardCache {
    std::vector<float> logits;         // (d_words)
    std::vector<float> hidden;         // (d_embed)   — post MLP local, pre unembedding
    std::vector<float> hidden_neurons; // (d_neurons) — post ReLU
    std::vector<float> mlp_input;      // (d_embed)   — emb[last] + attn_out (residual)
    std::vector<float> attn_weights;   // (seq_len)   — softmax output
    std::vector<float> Q_n;            // (d_qkv)     — query for last position
    std::vector<float> K_all;          // (seq_len × d_qkv)
    std::vector<float> V_all;          // (seq_len × d_qkv)
    std::vector<float> embs;           // (seq_len × d_embed)
    std::vector<int>   tokens;         // (seq_len)
};

void write_vec(std::ofstream& out, const std::vector<float>& v);
void save_weights(const WeightStorage& w, const char* path);
WeightStorage load_weights(const char* path, const ModelDimensions& d);

// Forward pass over a token sequence. Predicts next token after tokens.back().
// Requires d.d_qkv == d.d_embed (used for residual connection).
auto pass_word(std::vector<int> tokens, const WeightStorage& w, const ModelDimensions& d) -> ForwardCache;

// Computes gradients for all weight matrices.
// Gradient vector order: dW_e, dW_q, dW_k, dW_v, dW_g, dW_bg, dW_l, dW_bl, dW_u
auto compute_full_gradients(
    int target_token,
    const ForwardCache& cache,
    const WeightStorage& w,
    const ModelDimensions& d
) -> std::vector<float>;