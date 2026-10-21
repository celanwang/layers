# L7-T2 execution output

## Acceptance

Both section 5 commands exit 0. Tests: **156 passed**, one existing PyTorch
warning (27.30s). All RNG/token comparisons are bit-exact; trained checkpoint
T2/T1 logits are exact, max absolute error to L6 0.0. Sampling instrumentation
confirms one `[B]` eval and one list transfer, with no scalar host reads.

## Three-stage B=8 pairs

| Pair | L6 baseline tok/s | T1-only tok/s | T1+T2 tok/s | Within-pair gain |
|---|---:|---:|---:|---:|
| l6_vs_t1 | 1449.17 | 1645.67 | — | +13.56% |
| t1_vs_t2 | — | 1663.67 | 1918.45 | +15.31% |
| l6_vs_t2 | 1450.15 | — | 1891.05 | +30.40% |

T2 increment: **+15.31% throughput**, decode median
307.753 -> 266.882 ms
(**-13.28%** time). Prediction **supported**
(>=5% threshold). Independent combined L6/T2 gain:
**+30.40%**; >=1800 tok/s combined prediction supported.
Every ratio uses its own pair; repeated T1/L6 values are separate experiments.

Protocol: M4 Max / 48 GiB, MLX 0.32.4, fp32, unchanged L6 harness,
3 warmups / 10 repeats, paired alternating, B=1/2/4/8, 32-token prompts,
64 generated tokens. Config `08e7ca285efb`; implementation `2f81b785f4bddc64e2ecf4abeb3438997d9ca5f7`.
Raw samples/source hashes: `sampling-bench-20261021T155558005176Z.json`. Evidence verification checks all
720 raw timing values; all 37,440 paired token comparisons pass.

## Pitfalls and deviations

- Per-row seed/draw/categorical and probability semantics are retained, including
  greedy RNG consumption. Larger lazy sampling graphs still perform row work.
- Additive `BatchedSamplingEngine` uses the exact T1 KV calls. Custom callable
  batches use inherited T1 behavior; single-fence sampling applies to standard
  `Sampler` batches. Existing engines and `Sampler` are source-identical.
- Combined B=1 latency rises 4.03% vs L6;
  combined B=2 throughput also slightly regresses. Full small-batch results are
  retained; performance claims apply to the measured B=8 workload.
- Existing bf16 test's newly emitted `results/l5/` JSON was relocated intact
  into whitelisted `results/l7/`, following T1 precedent. Sole transient
  out-of-whitelist write; no old artifact, test, dependency or protocol change.
- No literal `-o` path supplied: this file is the section 6 / -o destination.
  No discarded benchmark runs, RNG/KV changes or new dependencies. No push.

## Actual changed files

- `bench/l7_sampling_bench.py`
- `layers/__init__.py`
- `layers/engine.py`
- `layers/inference.py`
- `results/l7/bf16-train-20261021T155410864907Z.json`
- `results/l7/l7-t2-output.md`
- `results/l7/l7-t2-report.md`
- `results/l7/sampling-bench-20261021T155558005176Z.json`
- `results/l7/t2-bench-acceptance.txt`
- `results/l7/t2-checkpoint-oracle.json`
- `results/l7/t2-evidence-verification.json`
- `results/l7/t2-scope-audit.json`
- `results/l7/t2-tests-acceptance.txt`
- `runs/l7-implementation-notes.md`
- `tests/test_l7_batched_sampling.py`

## Commits and report

`63f5789` records the plan; `2f81b78` contains the verified implementation;
the final archive commit contains this output and complete measurement/report.
Full report: [`l7-t2-report.md`](l7-t2-report.md).
