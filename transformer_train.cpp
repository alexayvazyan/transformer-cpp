#include "transformer.h"
#include "helpers.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <fstream>

int main() {
    ModelDimensions d;
    d.d_words   = 10000;
    d.d_embed   = 128;
    d.d_qkv     = 128;  // must equal d_embed for residual connection
    d.d_neurons = 256;

    const char* weights_path = "/Users/alexanderayvazyan/Documents/cpplearning/project/weights.bin";
    WeightStorage w = load_weights(weights_path, d);

    const float lr = 0.01f;
    std::vector<int> tokens = {5, 42, 17, 8};  // context window
    int next_token = 12;                         // target for position after tokens.back()

    auto cache = pass_word(tokens, w, d);
    auto gv    = compute_full_gradients(next_token, cache, w, d);

    // Apply gradients (order matches compute_full_gradients: dW_e, dW_q, dW_k, dW_v, dW_g, dW_bg, dW_l, dW_bl, dW_u)
    size_t off = 0;
    for (size_t i = 0; i < w.W_e.size();  ++i) w.W_e[i]  -= lr * gv[off++];
    for (size_t i = 0; i < w.W_q.size();  ++i) w.W_q[i]  -= lr * gv[off++];
    for (size_t i = 0; i < w.W_k.size();  ++i) w.W_k[i]  -= lr * gv[off++];
    for (size_t i = 0; i < w.W_v.size();  ++i) w.W_v[i]  -= lr * gv[off++];
    for (size_t i = 0; i < w.W_g.size();  ++i) w.W_g[i]  -= lr * gv[off++];
    for (size_t i = 0; i < w.W_bg.size(); ++i) w.W_bg[i] -= lr * gv[off++];
    for (size_t i = 0; i < w.W_l.size();  ++i) w.W_l[i]  -= lr * gv[off++];
    for (size_t i = 0; i < w.W_bl.size(); ++i) w.W_bl[i] -= lr * gv[off++];
    for (size_t i = 0; i < w.W_u.size();  ++i) w.W_u[i]  -= lr * gv[off++];

    save_weights(w, weights_path);

    int max_logit_idx = static_cast<int>(
        std::distance(cache.logits.begin(), std::max_element(cache.logits.begin(), cache.logits.end())));
    std::cout << "argmax(logits) = " << max_logit_idx
              << " (target = " << next_token << ")\n";

    std::cout << "First 10 non-zero gradients: ";
    int count = 0;
    for (size_t i = 0; i < gv.size() && count < 10; ++i) {
        if (gv[i] != 0.0f) {
            std::cout << gv[i];
            if (count < 9) std::cout << ", ";
            ++count;
        }
    }
    std::cout << "\n";

    return 0;
}


// Forward pass.
// Architecture: embeddings → single-head attention → residual → MLP(W_g+ReLU, W_l) → W_u → logits
// Attention output and embedding of last token are added (residual), then fed into the MLP.
// Requires d.d_qkv == d.d_embed for the residual addition to be dimension-compatible.
auto pass_word(std::vector<int> tokens, const WeightStorage& w, const ModelDimensions& d) -> ForwardCache {
    assert(d.d_qkv == d.d_embed && "d_qkv must equal d_embed for residual connection");
    assert(!tokens.empty());

    ForwardCache cache;
    cache.tokens = tokens;
    int seq_len = static_cast<int>(tokens.size());
    int last    = seq_len - 1;

    // --- Embeddings ---
    cache.embs.resize(seq_len * d.d_embed);
    for (int t = 0; t < seq_len; ++t) {
        const float* src = &w.W_e[tokens[t] * d.d_embed];
        float*       dst = &cache.embs[t * d.d_embed];
        for (int i = 0; i < d.d_embed; ++i) dst[i] = src[i];
    }

    // --- Q for last position, K and V for all positions ---
    cache.Q_n.assign(d.d_qkv, 0.0f);
    for (int i = 0; i < d.d_qkv; ++i)
        for (int j = 0; j < d.d_embed; ++j)
            cache.Q_n[i] += w.W_q[i * d.d_embed + j] * cache.embs[last * d.d_embed + j];

    cache.K_all.assign(seq_len * d.d_qkv, 0.0f);
    cache.V_all.assign(seq_len * d.d_qkv, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        for (int i = 0; i < d.d_qkv; ++i) {
            float k_sum = 0.0f, v_sum = 0.0f;
            for (int j = 0; j < d.d_embed; ++j) {
                k_sum += w.W_k[i * d.d_embed + j] * cache.embs[t * d.d_embed + j];
                v_sum += w.W_v[i * d.d_embed + j] * cache.embs[t * d.d_embed + j];
            }
            cache.K_all[t * d.d_qkv + i] = k_sum;
            cache.V_all[t * d.d_qkv + i] = v_sum;
        }
    }

    // --- Scaled dot-product attention (full context, no causal mask needed — predicting next token) ---
    float scale = 1.0f / std::sqrt(static_cast<float>(d.d_qkv));
    std::vector<float> scores(seq_len, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        float dot = 0.0f;
        for (int i = 0; i < d.d_qkv; ++i)
            dot += cache.Q_n[i] * cache.K_all[t * d.d_qkv + i];
        scores[t] = dot * scale;
    }
    cache.attn_weights = softmax(scores);

    // attn_out = sum_t(attn_weights[t] * V[t])
    std::vector<float> attn_out(d.d_qkv, 0.0f);
    for (int t = 0; t < seq_len; ++t)
        for (int i = 0; i < d.d_qkv; ++i)
            attn_out[i] += cache.attn_weights[t] * cache.V_all[t * d.d_qkv + i];

    // --- Residual: mlp_input = emb[last] + attn_out ---
    cache.mlp_input.resize(d.d_embed);
    const float* emb_last = &cache.embs[last * d.d_embed];
    for (int i = 0; i < d.d_embed; ++i)
        cache.mlp_input[i] = emb_last[i] + attn_out[i];

    // --- MLP global: mlp_input → neurons, bias + ReLU ---
    cache.hidden_neurons.resize(d.d_neurons);
    for (int i = 0; i < d.d_neurons; ++i) {
        float sum = w.W_bg[i];
        for (int j = 0; j < d.d_embed; ++j)
            sum += cache.mlp_input[j] * w.W_g[i * d.d_embed + j];
        cache.hidden_neurons[i] = std::max(0.0f, sum);
    }

    // --- MLP local: neurons → embed ---
    cache.hidden.resize(d.d_embed);
    for (int i = 0; i < d.d_embed; ++i) {
        float sum = w.W_bl[i];
        for (int k = 0; k < d.d_neurons; ++k)
            sum += cache.hidden_neurons[k] * w.W_l[k * d.d_embed + i];
        cache.hidden[i] = sum;
    }

    // --- Unembedding ---
    cache.logits.resize(d.d_words);
    for (int j = 0; j < d.d_words; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < d.d_embed; ++i)
            sum += cache.hidden[i] * w.W_u[i * d.d_words + j];
        cache.logits[j] = sum;
    }

    return cache;
}


auto compute_full_gradients(
    int target_token,
    const ForwardCache& cache,
    const WeightStorage& w,
    const ModelDimensions& d
) -> std::vector<float> {
    int seq_len = static_cast<int>(cache.tokens.size());
    int last    = seq_len - 1;

    // --- Cross-entropy loss → dz (gradient w.r.t. pre-softmax logits) ---
    auto probs = softmax(cache.logits);
    std::vector<float> dz(d.d_words);
    for (int j = 0; j < d.d_words; ++j) dz[j] = probs[j];
    dz[target_token] -= 1.0f;

    // --- Unembedding backward ---
    std::vector<float> dW_u(d.d_embed * d.d_words, 0.0f);
    std::vector<float> d_hidden(d.d_embed, 0.0f);
    for (int i = 0; i < d.d_embed; ++i) {
        for (int j = 0; j < d.d_words; ++j) {
            dW_u[i * d.d_words + j] = cache.hidden[i] * dz[j];
            d_hidden[i] += dz[j] * w.W_u[i * d.d_words + j];
        }
    }

    // --- MLP local backward ---
    std::vector<float> d_hidden_neurons(d.d_neurons, 0.0f);
    std::vector<float> dW_l(d.d_neurons * d.d_embed, 0.0f);
    std::vector<float> dW_bl(d.d_embed, 0.0f);
    for (int k = 0; k < d.d_neurons; ++k) {
        for (int i = 0; i < d.d_embed; ++i) {
            d_hidden_neurons[k]      += d_hidden[i] * w.W_l[k * d.d_embed + i];
            dW_l[k * d.d_embed + i]  = cache.hidden_neurons[k] * d_hidden[i];
        }
    }
    for (int i = 0; i < d.d_embed; ++i) dW_bl[i] = d_hidden[i];

    // ReLU backward
    for (int k = 0; k < d.d_neurons; ++k)
        if (cache.hidden_neurons[k] <= 0.0f) d_hidden_neurons[k] = 0.0f;

    // --- MLP global backward ---
    std::vector<float> d_mlp_input(d.d_embed, 0.0f);
    std::vector<float> dW_g(d.d_neurons * d.d_embed, 0.0f);
    std::vector<float> dW_bg(d.d_neurons, 0.0f);
    for (int i = 0; i < d.d_neurons; ++i) {
        dW_bg[i] = d_hidden_neurons[i];
        for (int j = 0; j < d.d_embed; ++j) {
            dW_g[i * d.d_embed + j]  = cache.mlp_input[j] * d_hidden_neurons[i];
            d_mlp_input[j]          += w.W_g[i * d.d_embed + j] * d_hidden_neurons[i];
        }
    }

    // --- Residual split: mlp_input = emb[last] + attn_out ---
    // Gradient flows equally to both branches.
    std::vector<float> d_embs(seq_len * d.d_embed, 0.0f);
    for (int i = 0; i < d.d_embed; ++i)
        d_embs[last * d.d_embed + i] += d_mlp_input[i];  // residual path to embedding

    std::vector<float> d_attn_out(d.d_qkv);
    for (int i = 0; i < d.d_qkv; ++i) d_attn_out[i] = d_mlp_input[i];  // attention path

    // --- Attention backward ---
    // attn_out = sum_t(attn_weights[t] * V[t])
    std::vector<float> dV_all(seq_len * d.d_qkv, 0.0f);
    std::vector<float> da(seq_len, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        for (int i = 0; i < d.d_qkv; ++i) {
            dV_all[t * d.d_qkv + i]  = cache.attn_weights[t] * d_attn_out[i];
            da[t]                    += cache.V_all[t * d.d_qkv + i] * d_attn_out[i];
        }
    }

    // Softmax backward: ds[t] = a[t] * (da[t] - dot(a, da))
    float sum_ada = 0.0f;
    for (int t = 0; t < seq_len; ++t) sum_ada += cache.attn_weights[t] * da[t];
    std::vector<float> ds(seq_len);
    for (int t = 0; t < seq_len; ++t)
        ds[t] = cache.attn_weights[t] * (da[t] - sum_ada);

    // Backprop through scores = Q_n · K[t] / sqrt(d_qkv)
    float scale = 1.0f / std::sqrt(static_cast<float>(d.d_qkv));
    std::vector<float> dQ_n(d.d_qkv, 0.0f);
    std::vector<float> dK_all(seq_len * d.d_qkv, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        for (int i = 0; i < d.d_qkv; ++i) {
            dQ_n[i]                  += scale * ds[t] * cache.K_all[t * d.d_qkv + i];
            dK_all[t * d.d_qkv + i]  = scale * ds[t] * cache.Q_n[i];
        }
    }

    // Backprop through V[t] = W_v @ emb[t]
    std::vector<float> dW_v(d.d_qkv * d.d_embed, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        for (int i = 0; i < d.d_qkv; ++i) {
            for (int j = 0; j < d.d_embed; ++j) {
                dW_v[i * d.d_embed + j]      += dV_all[t * d.d_qkv + i] * cache.embs[t * d.d_embed + j];
                d_embs[t * d.d_embed + j]    += w.W_v[i * d.d_embed + j] * dV_all[t * d.d_qkv + i];
            }
        }
    }

    // Backprop through K[t] = W_k @ emb[t]
    std::vector<float> dW_k(d.d_qkv * d.d_embed, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        for (int i = 0; i < d.d_qkv; ++i) {
            for (int j = 0; j < d.d_embed; ++j) {
                dW_k[i * d.d_embed + j]      += dK_all[t * d.d_qkv + i] * cache.embs[t * d.d_embed + j];
                d_embs[t * d.d_embed + j]    += w.W_k[i * d.d_embed + j] * dK_all[t * d.d_qkv + i];
            }
        }
    }

    // Backprop through Q_n = W_q @ emb[last]
    std::vector<float> dW_q(d.d_qkv * d.d_embed, 0.0f);
    for (int i = 0; i < d.d_qkv; ++i) {
        for (int j = 0; j < d.d_embed; ++j) {
            dW_q[i * d.d_embed + j]          = dQ_n[i] * cache.embs[last * d.d_embed + j];
            d_embs[last * d.d_embed + j]     += w.W_q[i * d.d_embed + j] * dQ_n[i];
        }
    }

    // Backprop through W_e: emb[t] = W_e[tokens[t]]
    // All sequence positions contributed to attention, so all accumulate into dW_e.
    std::vector<float> dW_e(d.d_words * d.d_embed, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        int tok = cache.tokens[t];
        for (int j = 0; j < d.d_embed; ++j)
            dW_e[tok * d.d_embed + j] += d_embs[t * d.d_embed + j];
    }

    // --- Flatten: dW_e, dW_q, dW_k, dW_v, dW_g, dW_bg, dW_l, dW_bl, dW_u ---
    std::vector<float> gv;
    gv.reserve(dW_e.size() + dW_q.size() + dW_k.size() + dW_v.size() +
               dW_g.size() + dW_bg.size() + dW_l.size() + dW_bl.size() + dW_u.size());
    gv.insert(gv.end(), dW_e.begin(),  dW_e.end());
    gv.insert(gv.end(), dW_q.begin(),  dW_q.end());
    gv.insert(gv.end(), dW_k.begin(),  dW_k.end());
    gv.insert(gv.end(), dW_v.begin(),  dW_v.end());
    gv.insert(gv.end(), dW_g.begin(),  dW_g.end());
    gv.insert(gv.end(), dW_bg.begin(), dW_bg.end());
    gv.insert(gv.end(), dW_l.begin(),  dW_l.end());
    gv.insert(gv.end(), dW_bl.begin(), dW_bl.end());
    gv.insert(gv.end(), dW_u.begin(),  dW_u.end());
    return gv;
}
