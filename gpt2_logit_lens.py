"""
GPT-2 Small — Logit Lens + Direct Logit Attribution
=====================================================
Three things demonstrated:
  1. run_with_cache  — single forward pass, activations at every hook point
  2. Logit lens      — what does each layer predict at a given token position?
  3. Direct logit attribution — which layers/components contribute to a specific token's logit?

The prompt is a factual completion: "The Eiffel Tower is located in the city of"
GPT-2 should predict " Paris". We watch the model build that prediction layer by layer.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer, utils

# ── Load model ────────────────────────────────────────────────────────────────

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}\n")

model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

# GPT-2 Small: 12 layers, d_model=768, 12 heads, d_head=64, d_mlp=3072


# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT = "The Eiffel Tower is located in the city of"
TARGET = " Paris"

tokens      = model.to_tokens(PROMPT)               # (1, seq_len)
str_tokens  = model.to_str_tokens(PROMPT)
target_id   = model.to_single_token(TARGET)

print(f"Prompt tokens: {str_tokens}")
print(f"Target token:  '{TARGET}'  (id={target_id})")
print()


# ── 1. run_with_cache ─────────────────────────────────────────────────────────
# A single forward pass that stores every intermediate activation.
# cache["blocks.{L}.hook_resid_post"] gives the residual stream after layer L.

logits, cache = model.run_with_cache(tokens)

# Top-5 predicted next tokens from the full model
top = logits[0, -1].topk(5)
print("── Full model top-5 predictions ──────────────────────────────────────")
for val, idx in zip(top.values, top.indices):
    tok = model.to_string(idx)
    marker = " ◀" if idx == target_id else ""
    print(f"  {tok!r:<12}  logit={val.item():+.2f}{marker}")
print()


# ── 2. Logit lens ─────────────────────────────────────────────────────────────
# For each layer L, take the residual stream at that point, apply the final
# layer norm and unembed, and read off what the model "would predict" if it
# stopped there.  This shows how the prediction builds across layers.

print("── Logit lens — top prediction and target rank at each layer ─────────")
col3 = '"Paris" logit'
col4 = '"Paris" rank'
print(f"  {'Layer':<8} {'Top token':<14} {'Top logit':>10}  {col3:>14}  {col4:>12}")
print(f"  {'─'*8} {'─'*14} {'─'*10}  {'─'*14}  {'─'*12}")

for layer in range(model.cfg.n_layers):
    resid = cache[f"blocks.{layer}.hook_resid_post"]   # (1, seq_len, 768)

    # Apply final layer norm + unembed to the last token position
    resid_final_norm = model.ln_final(resid)            # (1, seq_len, 768)
    layer_logits     = model.unembed(resid_final_norm)  # (1, seq_len, vocab)
    last_logits      = layer_logits[0, -1]              # (vocab,)

    top1_id    = last_logits.argmax().item()
    top1_tok   = model.to_string(top1_id)
    top1_logit = last_logits[top1_id].item()

    target_logit = last_logits[target_id].item()
    target_rank  = (last_logits > target_logit).sum().item() + 1

    marker = " ◀" if top1_id == target_id else ""
    print(f"  L{layer:<7} {top1_tok!r:<14} {top1_logit:>+10.2f}  {target_logit:>+14.2f}  {target_rank:>12}{marker}")

print()


# ── 3. Direct logit attribution ───────────────────────────────────────────────
# Decompose the final logit for TARGET into contributions from each component.
#
# The residual stream is a sum: embed + pos_embed + Σ_L (attn_L + mlp_L)
# Each component adds a vector to the stream. We can project each contribution
# onto the "Paris" direction in logit space to see who is responsible.
#
# logit(" Paris") = (W_U @ ln_final(resid_final))[target_id]
# We use the linear approximation: treat ln_final as roughly linear,
# so each component's contribution ≈ (W_U[target_id]) · component_output.
# This is an approximation but gives correct directional intuitions.

print("── Direct logit attribution — contribution to \" Paris\" logit ─────────")
print("   (approximate: ignores layer norm nonlinearity, directionally correct)")
print()

# The unembedding direction for our target token
unembed_dir = model.W_U[:, target_id]  # (d_model,) — the direction that votes for "Paris"

def logit_contrib(vec):
    """Project a residual stream vector onto the target unembedding direction."""
    # vec: (1, seq_len, d_model) or (1, d_model)
    v = vec[0, -1] if vec.ndim == 3 else vec[0]
    return (v @ unembed_dir).item()

# Embedding contribution
embed_contrib = logit_contrib(cache["hook_embed"] + cache["hook_pos_embed"])

print(f"  {'Component':<25} {'Contribution':>14}")
print(f"  {'─'*25} {'─'*14}")
print(f"  {'Embedding + PosEmbed':<25} {embed_contrib:>+14.3f}")

layer_contribs = []
for layer in range(model.cfg.n_layers):
    attn = logit_contrib(cache[f"blocks.{layer}.hook_attn_out"])
    mlp  = logit_contrib(cache[f"blocks.{layer}.hook_mlp_out"])
    layer_contribs.append((layer, attn, mlp))
    print(f"  {'L'+str(layer)+' attn':<25} {attn:>+14.3f}")
    print(f"  {'L'+str(layer)+' mlp':<25} {mlp:>+14.3f}")

print()

# Summary: top 5 positive and negative contributors
all_contribs = [("Embed+Pos", embed_contrib)]
for layer, attn, mlp in layer_contribs:
    all_contribs.append((f"L{layer} attn", attn))
    all_contribs.append((f"L{layer} mlp",  mlp))

all_contribs.sort(key=lambda x: -x[1])

print(f"  Top 5 components pushing TOWARD \" Paris\":")
for name, val in all_contribs[:5]:
    bar = "█" * int(abs(val) * 3)
    print(f"    {name:<15} {val:>+8.3f}  {bar}")

print(f"\n  Top 5 components pushing AWAY FROM \" Paris\":")
for name, val in all_contribs[-5:]:
    bar = "█" * int(abs(val) * 3)
    print(f"    {name:<15} {val:>+8.3f}  {bar}")

print()
print(f"  Sum of all contributions: {sum(v for _, v in all_contribs):+.3f}")
print(f"  Actual final logit:       {logits[0, -1, target_id].item():+.3f}")
print("  (gap is the layer norm nonlinearity — close enough for attribution)")
