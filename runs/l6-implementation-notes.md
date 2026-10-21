# L6-T1 implementation notes

## Design and blind implementation

The implementation uses only this project's L1–L5 components and the approved
L6 principle note. No inference engine or serving framework source was read.

- Add `BatchedEngine` without changing `Engine`'s request or decode behavior.
  This is a synchronous scheduler: `add_request` is the iteration-boundary
  admission/prefill operation, inherited from L4. Prompts prefill separately,
  including the existing prefix sharing and worst-case page reservation rules,
  then join the next `step` batch. Callers serialize admission and stepping.
- Sample in stable request order. Exclude terminal tokens from further model
  work, decode all continuing IDs as `[B, 1]`, distribute logits in that same
  order, and free completed sequences at the end of the iteration.
- Snapshot each sequence's real KV length before layer writes. Use it both as
  that row's RoPE position and as the inclusive attention key bound. Append
  real K/V to the unchanged page pool, gather, then right-pad to the batch max.
  Padding exists only in attention views and never consumes persistent pages.
- Mask scores with negative infinity before the explicit stable softmax.
  Inspect post-softmax probabilities directly in tests, including exact zeros.

## Assumptions and failure probes

Variable positions cannot use a single scalar RoPE offset. A separate batch
attention method makes the position broadcast explicit and preserves all old
operation paths. Tests compare absolute logit errors at `1e-5`, use both greedy
and seeded stochastic sampling, permute tokens together with their KV rows,
and exercise shared-page completion plus new arrivals.

Performance predictions remain hypotheses until the L0-protocol benchmark:
B=8 speedup at least 4x; B=1 latency regression below 10%. Negative results
will be retained alongside configuration and raw timing samples.

## Initial measurement and bounded optimization

The first uncompiled batch path passed all 97 tests. With 3 warmups and 10
paired repeats it measured B=8 at 1385.02 tok/s versus L4 at 346.94 tok/s:
3.992x, below the 4x gate. B=1 improved 15.42%. The full failed measurement
is retained in `results/l6/batching-bench-20261021T130956006308Z.json` and
`results/l6/bench-attempt1.txt`; the gate was not rounded up or weakened.

The batch path now compiles two pure arithmetic helpers (RoPE and explicit
masked attention) with `mx.compile`. Page operations remain inspectable Python;
the original single-request operations and all measurement settings are intact.
No `mx.fast` operation is used. We consulted only the installed public
`mx.compile` API docstring, without reading implementation sources.

The first compiled measurement passed at B=8 4.037x, with little margin (raw
record `batching-bench-20261021T131308951824Z.json`). A follow-up shape audit
made every dense projection explicitly rank two: attention Q/K/V/output,
MLP, and vocabulary projection operate on `[B, D]` rows. Attention still uses
`[B, H, 1, T]` and model inputs/outputs retain `[B, 1, ...]`. This directly
expresses shared-weight GEMM rather than relying on backend conversion of a
rank-three batch of one-row matrix multiplies. The L4 path is unchanged.

## Trained-checkpoint precision diagnosis

An additional check with the real width-512 checkpoint, eight variable prefixes,
and 33 sampled tokens found two absolute logit errors above `1e-5` out of
271,392 comparisons (max `1.1444091796875e-5`), although every token matched.
The failure is retained in `checkpoint-oracle-attempt1.json`. Disabling
compilation and changing dense layout produced the same two outliers
(`linear-layout-probes.json`), ruling out compilation as their source.

An fp64 CPU diagnostic decomposed each failing final logit into propagated
normalized-input drift and output-projection accumulation error. Input drift
was only `1.62e-6` / `-1.04e-6`; the batch vocabulary dot-product errors were
`8.39e-6` / `-1.02e-5` relative to fp64, versus reference GEMV errors below
`4.85e-7`. Thus the final long fp32 reduction dominated these failures.
`projection-precision-probe.json` preserves the decomposition and alternatives.

The batch vocabulary projection now sums two half-width GEMMs, shortening
the fp32 accumulation chains while continuing to share both weight halves
across all request rows. The diagnostic max absolute error fell to
`7.62939453125e-6`; no tolerance was relaxed. A trained-checkpoint regression
test covers the same eight prefixes and 32 actual decode forwards. CPU fp64
is diagnostic only; inference remains GPU fp32 throughout.

The unchanged bf16 training regression test automatically emits an L5 JSON.
Each newly generated artifact is relocated intact into `results/l6/` after the
suite; existing L5 files are neither edited nor removed.
