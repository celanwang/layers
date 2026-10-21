# L3-T1 Implementation Notes: GPT Module Assembly

## Model structure

- `LayerNorm` normalizes only the final feature axis with population variance, `eps=1e-5`, and learned `gamma` / `beta`.
- `MLP` uses bias-free `d_model → 4·d_model → d_model` projections and exact GELU, expressed with `erf`.
- `TransformerBlock` is pre-LN: `x += attention(LN(x))`, then `x += MLP(LN(x))`. It reuses `CausalSelfAttention` and drops its attention-weight return value.
- `GPT` accepts token ID arrays shaped `[sequence]` or `[batch, sequence]`, applies the configured stack and final LayerNorm, then computes logits with `hidden @ token_embedding.weight.T`. The output projection has no separate parameter.
- The model rejects sequences longer than `ctx`. The default configuration is 4 layers, width 512, 8 heads, context 256, and vocabulary size 1028 from the L1 tokenizer's 1024 ordinary IDs plus four reserved IDs.

## Initialization and loss

- Initialization starts with `mx.random.seed(42)` by default and draws the assembled model's weights from one seeded stream. This keeps layers distinct even though the reused L2 constructors seed themselves locally.
- Embeddings use normal weights scaled by `0.02`; attention and MLP projections use normal weights scaled by `1/sqrt(d_model)`, matching the L2 convention.
- `cross_entropy` computes log-sum-exp after subtracting the maximum logit, gathers the target logit, and returns the mean token loss. Optimizers, schedules, and clipping remain outside this L3-T1 task.

## Validation evidence

- The d_model=2, one-block hand case checks LayerNorm arithmetic, both residual paths with zeroed sublayers, tied logits, and the cross-entropy result.
- The two-layer PyTorch Oracle copies every named parameter through NumPy and compares logits, loss, and every named parameter gradient at `atol=rtol=1e-4`.
- Additional checks cover model-level causality, the exact default tied parameter count, text encoded by the L1 tokenizer flowing through GPT, and a 50-step inline SGD fixed-batch overfit pre-test.
- Full acceptance passed: `44 passed`.
