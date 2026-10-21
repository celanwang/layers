# L7 comparison: page materialization and one-fence sampling

L7 removes measurable per-request overhead, but still constructs contiguous KV
and row-shaped sampling graphs. The combined B=8 result is **1891.05 / 1893.00
tok/s** in the two direct L6/T2 pairs, leaving **32.28% / 32.21%** below the
historical 2792.31 tok/s projection. The evidence supports host small-operation
and synchronization overhead as a major limit on scaling in this workload;
it does not measure a host-time majority [S1, S2, L6, A].

### Evidence and measurement conventions

| ID | Existing evidence and exact location | Config hash |
|---|---|---|
| S1 | [T2 execution measurement][S1], `/throughput_scaling/{0,1,2,3}/pairs/{l6_vs_t1,t1_vs_t2,l6_vs_t2}` | `08e7ca285efb` |
| S2 | [Planning-side T2 rerun][S2], same pointers | `2647c88d1d41` |
| P1 | [Initial, uncompiled T1 measurement][P1], `/throughput_scaling` | `16e6ba63e56f` |
| P2 | [Compiled T1 measurement][P2], `/throughput_scaling` | `4e04eb84dc1c` |
| P3 | [Planning-side T1 rerun][P3], `/throughput_scaling` | `a7204f93b23d` |
| L6 | [Existing accounting][L6], `/gap`, `/padding`, `/scaling` | source `2d2e2aeed61b` |
| A | [Derived accounting for this analysis][A], `/materialization`, `/t2_models`, `/actual_scaling_padding`, `/int8_kv_scenario`, `/sensitivity` | `24ed91a0c873` |

S1/S2 and P1–P3 use Apple M4 Max / 48 GiB unified memory, MLX 0.32.4,
macOS 15.8, Python 3.12.15, fp32, the same 13,118,464-parameter checkpoint
(`a6bc079ef679c90ea8a4fdff95cf5086d4d49b8c811ce0252b2c31c7f1531889`),
4 layers / width 512 / 8 heads / head dimension 64. The unchanged harness uses
3 warmups, 10 measured repetitions, alternating paired order, 32-token distinct
prompts, 64 generated tokens, 63 decode forwards, block size 16, no sharing,
temperature=0. Median is sorted sample index 5; p90 is index 8. Decode includes
sampling, scheduling, KV work and synchronization, with prefill separate
([harness](../bench/l6_batching_bench.py), lines 21–29, 57–99, 108–131;
S1/S2 `/device`, `/protocol`; parameter count in [L5 analysis](l5-diff-analysis.md), §1).

All timing conclusions below reuse these archives. A records their hashes,
device/protocol/commit metadata and independently checks **2160 raw timing
values** and throughput arithmetic across the five L7 files. No performance
experiment or test suite was run for this documentation task. S1/S2's two edges
containing T2 remain separate series: **1918.45 / 1909.94** belongs to T1/T2,
not to the direct combined comparison [A `/sources`, `/t2_models`].

## 1. Structural gap: vectorized materialization and batched production kernels

**Conclusion.** T1 removes B separate prefix gathers, while retaining an
O(B × padded_length × width) output and an O(unique_pages × block_size × width)
compact bank; T2 shares one sampling fence while retaining per-row probability
and RNG graph construction. Neither change implements the complete production
kernel structure ([KV source](../layers/paged_kv.py), lines 382–424, 483–493;
[sampling source](../layers/inference.py), lines 184–210; [V1], [V3], [V4]).

T1 walks each row's block table in Python, deduplicates referenced physical
pages, concatenates their K/V payloads into one bank, then indexes both K and V
with one `mx.take`. A compiled transpose/reshape produces `[B,H,padded_length,D]`.
Page writes remain per row and layer ([engine](../layers/engine.py), 353–369;
[KV source](../layers/paged_kv.py), 308–424, 483–493). Compilation reduces
Python dispatch for the pure transform; it does not prove that concat, take,
layout conversion and attention lower to one Metal kernel.

For the requested **B=8, mean KV length 128** estimate, assume *equal current
lengths* of 128, no sharing, fp32 and H=8, D=64, S=16. There are 64 unique pages
and `padded_length=128`. The output size is
`C = 2 × 8 × 128 × 8 × 64 × 4 = 4,194,304 B = 4 MiB` per layer.
The bank is also `2 × 64 × 16 × 8 × 64 × 4 = 4 MiB` [A `/materialization`].

| Logical stage, per forward per layer | Payload written | Read + write accounting |
|---|---:|---:|
| Physical pages → compact K/V bank (`concatenate`) | 4 MiB | 8 MiB |
| Bank → batched selected K/V (`take`) | 4 MiB | 8 MiB |
| Attention consumes selected K/V | — | 4 MiB read |
| Total above, if stages materialize separately | 8 MiB | 20 MiB |
| Direct page-reading attention, necessary K/V input | — | 4 MiB read |

Across four layers the two copy stages write **32 MiB** and account for
**64 MiB read+write** per forward. These are graph-level byte budgets, not
measured DRAM traffic, peak allocations or exact copy/kernel counts. Caches,
compiler fusion and extra layout copies can change transactions. Block tables
are only 256 B, with 512 B of stacked K/V indices here. Mean length alone does
not determine max padding or unique pages: unequal lengths can increase output
and page rounding; shared prefixes reduce the bank but not duplicated logical
output rows [A; KV source, 398–419, 486–493].

PagedAttention §5.1 fuses reshape/block writes, page reads with attention, and
CoW block copies [PA]. The tagged CUDA kernel obtains physical page numbers
from each sequence's table and reads K/V at those addresses [V1, lines
198–270, 389–398]. Thus its “zero-copy” distinction means **no intermediate
contiguous-prefix reconstruction**: direct page-read attention eliminates the
bank and selected-prefix copy stages above, while still loading K/V and
writing attention output. It also does not by itself eliminate our Python
write/metadata loop; that needs the separate fused-write design [PA §5.1].
The v0.10.2 design page explicitly labels itself historical, so the tagged
kernel is the authority for current-at-that-tag details [V0 introduction; V1].

FlashAttention §3.1 / Algorithm 1 combines block statistics with a running
maximum and normalization sum, avoiding a global score/probability matrix [FA].
For example, merge `(m,l,u)` and block `(mb,lb,ub)` via
`m'=max(m,mb)`, `l'=exp(m-m')l+exp(mb-m')lb`,
`u'=exp(m-m')u+exp(mb-m')ub`, output `u'/l'`.
This is an exact mathematical route, not a claim that every PagedAttention
variant uses the identical FlashAttention schedule. V1 uses shared-memory
scores; V2 stores partition maxima/sums/output and reduces them [V1,
lines 298–351, 529–568, 638–660]. Our explicit compiled softmax remains
inspectable ([attention](../layers/attention.py), 278–285).

On sampling, T2 still loops over B samplers, invokes their probability code,
resets `seed+draws`, and constructs B scalar categorical draws. It stacks and
evaluates once, then transfers one token list; this preserves the scalar
oracle's RNG stream, including greedy draws ([inference](../layers/inference.py),
148–181, 184–210). A model iteration also retains `mx.eval(logits)` before row
splitting: **one sampling fence does not mean one fence for the whole step**
([engine](../layers/engine.py), 454–480).

The vLLM V1 processor interface applies to an entire logits tensor and updates
state as batch membership changes [V3; VD3]. Its sampler applies processors
by processor, broadcasts row temperatures, uses a batched argmax and returns
early for all-greedy batches [V4, lines 85–101, 128–176]. This differs from our
row-wise one-hot/log/categorical graphs even at temperature=0. Production
batch processing does not imply one universal fused sampling kernel, zero
Python work or identical RNG semantics; that would exceed these sources.

## 2. Remaining gap: a lower row slope with an increased intercept

**Conclusion.** The direct combined pair gives
`T(B)=1.8773+0.3025B` ms (S1) and `1.7955+0.3122B` ms (S2), compared with
historical L6's `1.6406+0.5022B`: the slope falls **39.75% / 37.83%**, while
the intercept rises **0.2367 / 0.1550 ms** [A `/old_l6_model`, `/t2_models`].

Use exactly the L6 endpoint definition:
`T(B) = 1000 × median_decode_seconds(B) / 63`,
`b=(T(8)-T(1))/7`, `a=T(1)-b`. This is endpoint interpolation, not least
squares or a measurement of disjoint host/GPU components. The window contains
64 sampling rounds but only 63 forwards; T is an equivalent cost, not an
isolated forward latency. All four B values enter the evidence and residual
check; interior points do not determine a or b [L6 `/gap/endpoint_affine_model`;
harness, 95–98, 127, 135–138].

T2 in the **direct L6/T2 edge**, S1 then S2 [A `/t2_models`, `edge=l6_vs_t2`,
`arm=t2`; original S1/S2 `/throughput_scaling/i/pairs/l6_vs_t2/t2/decode`]:

| B | S1 tok/s | S1 decode median / p90 ms | S1 T(B) ms | S2 tok/s | S2 decode median / p90 ms | S2 T(B) ms |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 466.04 | 137.328 / 139.067 | 2.1798 | 481.98 | 132.787 / 136.320 | 2.1077 |
| 2 | 755.33 | 169.463 / 173.955 | 2.6899 | 772.99 | 165.592 / 168.831 | 2.6284 |
| 4 | 1310.10 | 195.405 / 202.001 | 3.1017 | 1320.99 | 193.794 / 196.901 | 3.0761 |
| 8 | 1891.05 | 270.749 / 272.826 | 4.2976 | 1893.00 | 270.469 / 273.201 | 4.2932 |

Separate fits for each edge, with no pooling across pairs [A `/t2_models`]:

| Source / edge / arm | a ms | b ms/row | Δa vs historical | Slope reduction vs historical |
|---|---:|---:|---:|---:|
| Historical L6 | 1.6406 | 0.5022 | — | — |
| S1 L6/T2, L6 control | 1.5940 | 0.5013 | −0.0465 | 0.18% |
| S1 L6/T2, T2 | 1.8773 | 0.3025 | +0.2367 | 39.75% |
| S2 L6/T2, L6 control | 1.5976 | 0.4963 | −0.0430 | 1.16% |
| S2 L6/T2, T2 | 1.7955 | 0.3122 | +0.1550 | 37.83% |
| S1 T1/T2, T1 | 1.7781 | 0.3884 | +0.1375 | 22.66% |
| S1 T1/T2, T2 | 1.8273 | 0.3011 | +0.1867 | 40.04% |
| S2 T1/T2, T1 | 1.7236 | 0.4001 | +0.0830 | 20.32% |
| S2 T1/T2, T2 | 1.8605 | 0.2993 | +0.2199 | 40.39% |

Within the direct combined pairs themselves the slope decreases 39.65% / 37.10%
and a increases 0.2832 / 0.1979 ms. Within T1/T2 the slope decreases 22.47% /
25.19%. These independent contrasts support the scaling change without
pretending that historical drift is a component effect [A, corresponding fits].

Direct-pair residuals `actual−(a+bB)` at B=2/4 are **+0.2075/+0.0142 ms**
(S1) and **+0.2085/+0.0318 ms** (S2); endpoints have zero residual by
construction. The B=2 residual is about 7.7–7.9% of measured T(2), so this
affine model misses a repeatable small-batch shape. At B=8 the algebraic row
term remains **56.32% / 58.18%** of T(8); that fraction is not a host-time
percentage [A `/t2_models/.../rows`, `/b8_row_term_fraction`].

The historical target `8 × 349.039 = 2792.311 tok/s` is an L4 projection,
not measured vLLM throughput or a physical upper bound. Its equivalent window
is 183.361 ms. The direct combined windows remain **87.388 / 87.109 ms**
above it, requiring **32.28% / 32.21% less current time** to reach it
(or **47.66% / 47.51% more throughput**). The T1/T2 edge's 1918.45 / 1909.94
leaves 31.30% / 31.60% below the same target. Keep these denominators explicit
[L6 `/gap`; A `/t2_models`].

Residual candidates, ranked by evidence rather than invented time shares:

| Candidate | Specific surviving work and evidence | Attribution strength |
|---|---|---|
| **Page writes, Python metadata and bank construction** | B=8 × 4 layers = **32** per-row `append_pages` calls/forward; fixed-size tail pages still concatenate prefix/new/suffix; each layer revisits referenced tables to build its bank. At equal L=128/no sharing, **256 page-table visits** across four layers, plus the §1 copy budget ([engine](../layers/engine.py), 364–369; [KV](../layers/paged_kv.py), 336–419). | Strongest structural candidate group; T1's B=8 improvement shows related old work mattered. No post-T2 ablation separates writes, lookup, bank copying or layout cost. |
| **Padded attention arithmetic** | `q@k.T` and `probs@v` run over every padded slot; the mask makes probabilities zero, not GEMMs sparse ([attention](../layers/attention.py), 278–285). In this equal-prompt workload current lengths are 33…95; block rounding adds **11.90% key slots**, or 10.64% of padded slots [A `/actual_scaling_padding`]. | Real extra work, unlikely the largest explanation. L6's uneven-length comparison paid only 2.40% throughput for much more padding, and its attention matmul FLOP share was 3.84%; neither number is a post-T2 wall-time bound [L6 `/padding`]. |
| **Scheduler and surviving row sampling graphs** | Active-request scan, events, generated-token bookkeeping, completion/free logic and B probability/seed/categorical constructions remain ([engine](../layers/engine.py), 448–485; [inference](../layers/inference.py), 202–210). | Source-confirmed O(B) work; T2's direct +15.31%/+15.74% B=8 gain concerns shared sampling boundaries, not the scheduler's isolated share [S1/S2 `t1_vs_t2`]. HTTP/SSE is absent from this harness. |
| **`logits[row,0]` split and next-step restack** | B MLX indexing calls after the logits eval, then `mx.stack` at next sampling entry ([engine](../layers/engine.py), 454–456, 478–480). | **Weakest cost evidence.** Calls exist, but no timing/trace shows material copies or significant stalls. Indexing can be views; there is no per-row `.item()` here. |

The largest remaining *candidate* is therefore page write/metadata/bank work
together with row graph dispatch, not a proved largest timed component. Even
dense projections vary with B; the endpoint slope cannot identify host work
exclusively (shared model arithmetic: [engine](../layers/engine.py), 378–392).

## 3. Small-batch regressions and path selection

**Conclusion.** Fixed bank/graph work is poorly amortized at B=1/2 and short
histories; observed signs drift, so there is no measured universal crossover
or zero-regression guarantee [P1–P3; S1/S2; §2 residuals].

Per-pair **throughput** changes versus the same pair's L6 control
(S1/S2 `/throughput_scaling/{0,1}/pairs`):

| Path | S1 B=1 | S1 B=2 | S2 B=1 | S2 B=2 |
|---|---:|---:|---:|---:|
| T1 only, `l6_vs_t1` | −5.13% | −1.70% | −3.33% | +0.98% |
| Combined, `l6_vs_t2` | −3.88% | −0.88% | −0.65% | +3.63% |

Standalone T1 B=1 **latency** increases are **8.75%** (P1, uncompiled),
**1.32%** (P2, compiled), **8.43%** (P3, compiled rerun).
P2→P3 already demonstrates drift with unchanged materialization source;
P1→P2 also includes a code change and cannot isolate compilation's gain.
Combined S1 B=1 latency is +4.03%, which corresponds to −3.88% throughput,
not −4.03% throughput [A `/t1_regressions`; P1–P3 `/single_request_latency`;
S1 `l6_vs_t2`]. The specification's approximate −0.9%…−5.1% range mixes paths
and omits S2's positive B=2 results; all are retained here.

The L7 path builds a dedup dictionary/table, a bank, combined K/V indices,
take and layout expressions even for B=1. It still writes the same page payload
as L6. The old gather concatenates only the few live pages at L=33…95 (3…6
pages); for a single page it can return that payload directly. Consequently
removing B row gathers saves little at small B/L, whereas bank construction
has a fixed graph/dispatch cost and copies data. This is a mechanism consistent
with the regressions, not a timed allocation of them [KV, 336–424, 426–454,
483–493; A `/actual_scaling_padding`].

The helper is compiled and keyed by array shapes/page-input structure;
the workload also crosses page boundaries, changing the graph's shape.
Warmups exist, so cold compilation alone cannot be blamed. Without a compile
or GPU trace we cannot distinguish dispatch, specialization and layout costs
from thermal/scheduling noise [P1–P3 `/protocol`; KV, 483–493].

vLLM does have comparable shape-dependent **kernel selection**: the v0.10.2
`PagedAttention.forward_decode` API/source uses partitions of 512, chooses V1
if `max_seq_len<=8192` and either there is one partition or
`num_seqs*num_heads>512`; otherwise it chooses V2. Its comments identify
reduction overhead versus available parallelism/shared-memory limits
[VD2; V2, lines 20–21, 123–174]. Both paths read paged KV: this is not evidence
for a production dense-gather fallback at B≤2. An analogous local B/length
dispatch is a plausible later experiment, requiring a crossover grid,
mixed-length cases, numerical alignment and maintenance of two valid paths;
the current sparse grid does not justify a threshold or an implementation change.

## 4. Hypothesis verdict and its boundary

**Conclusion.** Controlled changes support “host small operations and
synchronization are a major scaling constraint for this local small-model
decode workload”; they do not establish “CPU/host accounts for most remaining
time” [S1/S2, §2].

T1 changes KV reconstruction while preserving sampling; T2 keeps T1 decode and
RNG semantics while merging sampling eval/transfers. S1's B=8 pairs give
T1 +13.56%, T2 increment +15.31%, combined +30.40%; S2 gives +11.84%,
+15.74%, +29.70%, respectively. The combined time reductions are 23.32% /
22.90%. Each percentage comes from its own pair, without chaining different
T1/L6 observations [S1/S2 `/throughput_scaling/3/pairs`; [T1 report](../results/l7/l7-t1-report.md),
“Retained first attempt”; [T2 report](../results/l7/l7-t2-report.md),
“Implementation and comparison with the immutable oracle”].

The interventions also change device graphs, copy scheduling, allocation and
opportunity for compiler fusion. Thus their combined delta is not a clean
CPU-time counter, and the model's 56–58% row term is not proof of host
dominance. The old principle's “decisive” intent becomes this bounded verdict,
with the following limits ([principle](l7-decode-perf-principle.md), §§1–5;
S1/S2; KV 483–493; inference 202–210):

| Cannot claim | Missing evidence and next tightening step |
|---|---|
| A measured majority of CPU time, exact Metal launch/stall counts, memory bandwidth saturation, or zero copies | **No Metal trace.** Correlate CPU graph building/eval waits with GPU kernels, copy/layout events, queue gaps and critical-path duration; short capture plus CPU profiling is needed, not just summing overlapping GPU events [M2; §1 graph-vs-traffic distinction]. |
| The same performance gain under top-k/top-p/random sampling | Timing uses **temperature=0 only**. Stochastic token/RNG correctness already has evidence, but not stochastic timing; repeat paired timing under mixed random workloads, preserving request RNG semantics [S1/S2 `/protocol`; [T2 report](../results/l7/l7-t2-report.md), “Acceptance and coverage”]. |
| The same bottleneck in larger production models or long contexts | Only **one 13M model**, one short equal-prompt workload and fp32 are timed here. Increase model/sequence/batch sizes under a separate approved recipe and report whether projection/KV costs overtake host work [harness 57–58, 82–99]. |
| CUDA/multi-GPU/discrete-memory performance or vLLM throughput parity | Only **one unified-memory M4 Max device** was measured; CUDA sources establish structural differences. Repeat on the target hardware to make transfer/communication claims [S1/S2 `/device`; PA §6.1]. |

The highest-value next observation is whether the eliminated host waits/queue
gaps account for most of the measured wall-time delta and what now occupies the
critical path. That would narrow the causal explanation; randomized sampling
and a larger model would separately narrow its generalization boundary.

## 5. Follow-up special topics: cost, benefit and suggested layer

**Conclusion.** Prioritize a short L7 profiling closure, then an L7 fused-SDPA
ablation; return compressed page integration to a distinct L5×runtime task
after the remaining conversion/synchronization cost is understood [§§1–4;
L5 analysis §2]. These are proposals for Kimi's decision, not new work started
by this analysis.

All throughput estimates below are **conditional planning estimates**, not
measured forecasts. A uses Amdahl's expression
`gain = 1 / ((1-f)+f/s) − 1`, where f is the current time fraction of the
targeted component and s its speedup. Neither fraction is supplied by FLOP
share, byte ratio or the endpoint fit [A `/sensitivity`; §2].

| Topic and proposed layer | Feasibility and evidence | Benefit magnitude / effort |
|---|---|---|
| **(a) L7 evidence closure: Metal capture plus system profiling; first priority** | Installed public `mx.metal.start_capture(path)` / `stop_capture()` exist. The MLX Metal Debugger guide requires `MTL_CAPTURE_ENABLED=1` and supports Xcode replay; optional debug builds improve labels [M2]. Capture warmed B=1/8 L6/T1/T2 windows separately, align CPU stacks/eval waits with GPU activity, and keep trace overhead separate from the unchanged timing baseline. | Profiling alone yields **0% runtime speedup**. It can account for the roughly **1.3 ms/forward-equivalent** direct combined reduction and the residual ~4.3 ms (S1/S2 B=8). If it exposes a removable 5–10% critical-path fraction, the theoretical follow-up gain is **5.3–11.1%** for complete removal; this is a sensitivity range, not a profile result. **0.5–1 day** including tool setup/interpretation; favor the installed capture API before considering an optional debug build. |
| **(b) L7 kernel ablation: `mx.fast.scaled_dot_product_attention` on T1 materialized KV; second priority** | Public MLX 0.32.4 signature accepts Q `[B,H,1,D]`, K/V `[B,H,Lpad,D]`, boolean or additive masks broadcastable to scores, and `force_fused` [M1]. Pass `valid[:,None,None,:]` for per-row lengths. `mask="causal"` alone includes padded future slots at q_len=1; it is insufficient. The API takes dense arrays, not page tables, so **bank/take copies remain** [§1]. | Replacing explicit attention can reduce attention intermediates/dispatch, while leaving per-row writes and sampling unchanged. If attention is 5–10% of current time and doubles, overall gain is **2.6–5.3%** (~49–100 tok/s at 1893); 20% time at 2× gives **11.1%**, a sensitivity case. Default to expecting a few percent or no gain in this small workload until measured. **0.5–1 day** for a whitelisted ablation, variable lengths/page tails, numerical oracle and paired timing. `force_fused=True` can reject unsupported shapes or choose a slower kernel: mask support does not guarantee fast execution [M1]. |
| **(c) Separate L5×runtime integration: INT8 pages and T1 consumption; after profiling** | Existing `QuantKVCache` is contiguous, not a paged-pool adapter; `get()` reconstructs fp32 history. Its quantizer performs host scalar validation and explicit evals ([inference](../layers/inference.py), 70–121; [quant](../layers/quant.py), 29–68, 179–201). A naive adapter would reintroduce synchronization. Compare **dequantize pages before building a fp32 bank** with **gather packed pages/scales then dequantize**, and ultimately **read/dequantize packed pages inside attention**. The last requires a new page-aware kernel and scale/CoW semantics; dense SDPA alone cannot consume packed pages. | At B=8/L=128, persistent fp32 KV is **16 MiB** across four layers; INT8 plus fp16 per-token K/V scales is **4.015625 MiB**, saving **11.984375 MiB**, or **3.9844×** capacity [A `/int8_kv_scenario`]. Dequant-before-bank retains the two 4 MiB/layer fp32 temporaries and adds conversion; no positive speedup is defensible. Packed gather lowers bank/gather payload ~4× but still creates fp32 attention input. A fused compressed reader could reduce KV input ~4×; if KV movement is 5–10% of time, a 4× component speedup gives only **3.9–8.1%** overall (~74–154 tok/s), before quantization overhead. **1–2 days** for a safe prototype; **3–5 days** for kernel/CoW/oracle/quality integration, subject to narrowing scope. |

L5's existing evidence is a concrete warning against predicting “4× storage
means 4× decode”: INT8 decode was **0.292–0.337×** fp32, with 4.6–5.2 ms/token
extra in the old single-request path. Its component ablation found
quantize-new/check/synchronization much larger than dequant-only ([L5 analysis](l5-diff-analysis.md),
§2, especially lines 63–93; [L5 report](../results/l5/l5-t2-report.md),
“KV memory and speed”). Those measurements are not a post-T2 B=8 forecast.
Any compressed-page task should preserve scale storage and quality oracles,
measure capacity and latency separately, and explicitly avoid restoring the
legacy scalar-check/eval pattern. No integration or scope expansion is claimed here.

### External references read for this task

All vLLM source links below use **v0.10.2 commit
`01efc7ef781391e744ed08c3292817a773d654e6`** (verified via GitHub's public
tag API). Only references and our analysis are archived; no external source is
copied into this repository. Read on 2026-10-22.

| ID | Public location and part read |
|---|---|
| PA | [PagedAttention paper][PA], arXiv:2309.06180v1, **§5.1** fused writes/read+attention/CoW, §4.6 control message and §6.1 hardware. |
| FA | [FlashAttention paper][FA], arXiv:2205.14135v2, **§3.1, Algorithm 1**, running normalization and fusion; Appendix B.1 recurrence. |
| V0 | [vLLM v0.10.2 Paged Attention design][V0], introduction's historical warning; Inputs, Key, Softmax, Value. |
| V1 | [Tagged `attention_kernels.cuh`][V1], **198–270, 298–351, 389–398, 529–568, 638–660**; page addressing and partition normalization. |
| V2 / VD2 | [Tagged `paged_attn.py`][V2], **20–21, 123–195**; [v0.10.2 API `forward_decode`][VD2], same heuristic. |
| V3 / VD3 | [Tagged processor `interface.py`][V3], **59–97**; [v0.10.2 API `LogitsProcessor`][VD3], `apply`, `update_state`, `is_argmax_invariant`. |
| V4 | [Tagged V1 `sampler.py`][V4], **85–117, 128–192**, processor execution, temperature, greedy/random branches. |
| M1 | Installed **MLX 0.32.4 public `mx.fast.scaled_dot_product_attention.__doc__`**, signature, shape/mask/precision/`force_fused`; [online API][M1] matches. |
| M2 | Installed public **`mx.metal.start_capture.__doc__` / `stop_capture.__doc__`**; [MLX Metal Debugger guide][M2], capture prerequisites, replay and optional debugging labels. |

### Acceptance and scope

The five numbered analysis points are complete, with byte/model accounting in
§§1–2 and explicit layer/benefit/effort proposals in §5. Derived arithmetic,
source hashes, local evidence links and the modification whitelist are checked
in [acceptance record](../results/l7/diff-acceptance.json).
Section 6 / `-o` output is [l7-diff-output.md](../results/l7/l7-diff-output.md);
no literal output filename was supplied, so this follows the T1/T2 convention.
No implementation, test, existing benchmark, dependency, protocol or historical
evidence file was changed. No new benchmark was needed. Tests are intentionally
not rerun, as task §5 directs. No push.

[S1]: ../results/l7/sampling-bench-20261021T155558005176Z.json
[S2]: ../results/l7/sampling-bench-20261021T160258087804Z.json
[P1]: ../results/l7/paged-attn-bench-20261021T154156026596Z.json
[P2]: ../results/l7/paged-attn-bench-20261021T154404587918Z.json
[P3]: ../results/l7/paged-attn-bench-20261021T154844862268Z.json
[L6]: ../results/l6/diff-accounting.json
[A]: ../results/l7/diff-accounting.json
[PA]: https://arxiv.org/html/2309.06180v1#S5
[FA]: https://arxiv.org/html/2205.14135v2#S3.SS1
[V0]: https://docs.vllm.ai/en/v0.10.2/design/paged_attention.html
[V1]: https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/csrc/attention/attention_kernels.cuh#L198-L270
[V2]: https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/attention/ops/paged_attn.py#L123-L195
[VD2]: https://docs.vllm.ai/en/v0.10.2/api/vllm/attention/ops/paged_attn.html#vllm.attention.ops.paged_attn.PagedAttention.forward_decode
[V3]: https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/sample/logits_processor/interface.py#L59-L97
[VD3]: https://docs.vllm.ai/en/v0.10.2/api/vllm/v1/sample/logits_processor/interface.html#vllm.v1.sample.logits_processor.interface.LogitsProcessor
[V4]: https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/sample/sampler.py#L85-L192
[M1]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html
[M2]: https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html
