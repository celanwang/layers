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
