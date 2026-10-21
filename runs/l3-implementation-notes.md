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

## L3-T2 Implementation Notes: Manual Training Loop

### Implementation

- `layers/optimizer.py` implements AdamW over nested MLX parameter trees. It keeps first and second moments by parameter path, applies bias correction, and applies decoupled weight decay directly to each parameter. Gradient clipping computes one global L2 norm over all leaves. `cosine_lr` uses one-based update numbers, reaches the peak at warmup step 100, then decays to 10% at update 200.
- `scripts/train.py` loads the L1 1024-token BPE vocabulary, encodes the whole Shakespeare file, and splits the token stream at `len(tokens) * 9 // 10`. Training shuffles non-overlapping 256-token windows from the training partition; heldout evaluation uses a fixed sample from the separate heldout partition. The script saves the final parameter arrays, the 10-step loss curve, and JSON metrics.
- The selected training configuration is fp32, 4 layers, width 512, 8 heads, context 256, batch 16, 200 updates, peak LR `3e-4`, 100-step warmup, betas `(0.9, 0.95)`, epsilon `1e-8`, clip norm `1.0`, and weight decay `0.1`. The specification did not set the numeric weight decay; the recorded run uses `0.1`.
- `tests/test_optimizer.py` checks three hand-computed AdamW updates, five updates against `torch.optim.AdamW` at `1e-6`, schedule boundaries, global norm clipping, and a 50-step fixed-batch overfit check. No training framework optimizer is used in the main path.

### Validation and Results

- `python -m pytest tests/ -q`: **49 passed**. The AdamW Oracle and the 50-step single-batch check passed; the latter reached `0.000338` nats from `2.763561`.
- Full-config benchmark: 3 warmups, 10 measured updates, 4,096 tokens/update; median `62,505.4` and p90 `62,762.6` tokens/s, config hash `7997ac760223`.
- The 200-step run reduced the sampled heldout loss from `7.078921` to `4.975763` nats (`29.71%`). The curve is finite and decreases throughout, but the task's strict `>30%` acceptance threshold was not reached. The command therefore exits nonzero and the training acceptance is **not passed**. Four attempt curves and metric files are retained in `results/l3/` to preserve the observed results.
- The model has `13,118,464` parameters. The 4× parameter-state estimate is `209,895,424` bytes; measured peak Metal allocation was `2,745,532,676` bytes (`2,618.34 MiB`), above the estimate plus the predicted `<1 GB` activations.
- Using the required formula and the L0 `14.75 TFLOP/s` reference, measured median throughput gives MFU `33.30%`, above the predicted `<10%`.
- Heldout tokens/byte is `0.418925`; model bpb is `3.0073`. The L1 uniform-ID proxies are `4.19` for layers and `4.69` for cl100k_base. The model bpb is lower because it measures a trained next-token model; the proxies assign uniform cost to token IDs and are not a like-for-like learned-model score.

### Acceptance Deviation

The mandated 200-step schedule and model configuration produce a clear finite loss decline, but not the specified 30% heldout reduction. Switching from random windows with replacement to shuffled full training windows improved the result to `29.71%`; the same sampler with weight decay `0.01` produced `29.73%` on its fixed sample. Both remain below the gate. No heldout data was used for gradient updates. Resolving this remaining gate requires a planning decision to revise the short-run threshold or authorize more training; this implementation preserves the failing result rather than changing the criterion.
