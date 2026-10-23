# Task: local inference service v1

Approved by the user on 2026-10-23 after the read-only implementation plan.

## Research question and risks

Can the existing numerical paths become a reproducible local service with
bounded generation, visible health, strict artifact loading and bounded drain?
Risks: leaking stop prefixes, confusing EOS with length, cross-request state,
silent partial checkpoints, shutdown leaks and measurement overhead.

## Allowed paths

`server/app.py`, `server/runtime.py`, `server/completion.py`,
`server/observability.py`, `tests/test_service_*.py`,
`server/model-manifest.json`,
`bench/final_service_bench.py`, `README.md`, `requirements-runtime.txt`,
`requirements-dev.txt`, `tasks/service-finalize.md`,
`runs/service-finalize-*.md`, `results/service-finalize/`.

## Prohibitions

No new dependencies or version changes, no existing test changes, no changes
to numerical model/engine paths or previous protocols/artifacts, no deletion,
push, external exposure or unrelated work. Preserve existing untracked data.

## Implementation contract

- Strict checkpoint key/shape/fp32/finite and tokenizer-ID-space checks before
  startup; record artifact hashes. Explicit L6/L7 CLI selection, L6 default.
- Automatic EOS suppression; optional 1..4 nonempty stop strings, at most 256
  characters each. Incremental UTF-8 and stop-prefix buffering; earliest full
  match terminates, stop text is excluded. Completion usage counts sampled
  tokens, including the token that triggers termination. Existing queue tuple
  remains `(content, finished, count)`; finish reason lives on the job.
- Local health/readiness/JSON metrics, bounded aggregate timing summaries,
  request-ID structured outcome logs without prompt/completion content.
- Drain rejects new jobs, completes accepted work until a deadline, then
  cancels/reclaims it. Hard close remains available for existing callers.
- Existing public call signatures remain backward compatible. Strictly local
  deployment, documented setup/checkpoint recovery and HTTP examples.

## Acceptance

`python -m pytest tests/ -q` using `.venv/bin/python`; syntax and whitespace
checks. New independent tests for stop/UTF-8, EOS, malformed assets, L7 parity,
JSON/SSE finish reasons, health, counters, cancellation and drain/reclamation.
`python bench/final_service_bench.py` records B=1/4/8 direct/API paired timing,
3 warmups, 10 observations, median/p90, raw samples, hashes and environment.
Preserve the existing <5% API-overhead target; retain failures and explain
protocol differences rather than changing a gate. No absolute throughput gate.

## Report

List files, commands/results, measured overhead, limitations and deviations;
principle, implementation notes, comparison and retrospective archived under
`runs/`. Commit each verified stage with conventional concise messages.
