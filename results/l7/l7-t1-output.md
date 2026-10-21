# L7-T1 execution output

Implemented additive page writes, single-index block-table materialization,
length-masked attention and `PagedAttentionEngine`. The L6 oracle and sampling
path remain unchanged. No framework attention/kernel implementation was read.

Validation: section 5 commands both exit 0; **133 tests pass** (119 existing
unchanged + 14 new), one existing warning. D512 checkpoint max absolute logit
error **0.0**, seeded tokens identical. Scope audit: 53 methods unchanged,
30 files byte-identical, only whitelist changes. Syntax/whitespace checks pass.

Final paired measurement (3 warmups/10 repeats, imported L6 harness,
config `4e04eb84dc1c`, implementation `e5b5755002e6`):

| B | L6 tok/s | L7 tok/s | Paired ratio | L7 / historical same-B L6 |
|---|---:|---:|---:|---:|
| 1 | 482.71 | 476.41 | 0.9869x | 1.0049x |
| 2 | 740.84 | 725.73 | 0.9796x | 1.0011x |
| 4 | 1109.55 | 1178.33 | 1.0620x | 1.0956x |
| 8 | 1421.73 | 1605.61 | 1.1293x | 1.1178x |

B=8 is 1.1178x historical L6 1436.41 tok/s.
Prediction: **intermediate**; >=1700 not reached, <1550 setback not triggered.
B=1 latency **+1.32%**: strict zero regression not proven.

Pitfalls: bank indexing adds copies/host operations, so singleton and B=2 can
regress even when larger batches improve. One bounded compile change was
made; both measurements and all raw samples are retained. New decode still
materializes O(B * padded_length) tensors and performs per-row page writes.
T2 sampling and Metal tracing remain outside scope. The existing bf16 test's
new outputs were relocated intact from results/l5 to results/l7. No literal
-o destination was supplied, so this is the execution summary destination.

Full report: `results/l7/l7-t1-report.md`.
Final evidence: `results/l7/paged-attn-bench-20261021T154404587918Z.json`. Initial evidence: `results/l7/paged-attn-bench-20261021T154156026596Z.json`.
Stage commits: `ad87130`, `fbbe38e`, `a266903`, `e5b5755`; final archive follows.
No push.

Actual changed files:

- `bench/l7_paged_attn_bench.py`
- `layers/__init__.py`
- `layers/attention.py`
- `layers/engine.py`
- `layers/paged_kv.py`
- `results/l7/bench-acceptance.txt`
- `results/l7/bench-compiled-materialization.txt`
- `results/l7/bf16-train-20261021T153617876397Z.json`
- `results/l7/bf16-train-20261021T154102370179Z.json`
- `results/l7/bf16-train-20261021T154306713377Z.json`
- `results/l7/checkpoint-oracle.json`
- `results/l7/evidence-verification.json`
- `results/l7/l7-t1-output.md`
- `results/l7/l7-t1-report.md`
- `results/l7/paged-attn-bench-20261021T154156026596Z.json`
- `results/l7/paged-attn-bench-20261021T154404587918Z.json`
- `results/l7/scope-audit.json`
- `results/l7/tests-acceptance.txt`
- `results/l7/tests-compiled-materialization.txt`
- `runs/l7-implementation-notes.md`
- `tests/test_l7_paged_attention.py`
