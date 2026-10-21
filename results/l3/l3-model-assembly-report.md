# L3-T1 Model Assembly Report

## Changed files

- `layers/transformer.py` — LayerNorm, MLP, and pre-LN TransformerBlock.
- `layers/model.py` — GPT assembly and stable cross-entropy.
- `layers/__init__.py` — public exports for the new module group.
- `tests/test_model.py` — hand, PyTorch Oracle, causality, parameter count, end-to-end, and overfit checks.
- `runs/l3-implementation-notes.md` — implementation decisions and validation notes.
- `results/l3/l3-model-assembly-report.md` — this execution report.

## Acceptance command

```text
$ /Users/wangchen/Developer/layers/.venv/bin/python -m pytest tests/ -q
............................................ [100%]
44 passed in 4.03s
```

## Parameter count

| Count | Value |
|---|---:|
| Measured unique stored parameters | 13,118,464 |
| Theoretical: `vocab·d + layers·(12·d² + 4·d) + 2·d` | 13,118,464 |
| Difference | 0 |

The vocabulary is 1028 (1024 tokenizer vocabulary entries plus four reserved IDs). The formula counts the tied embedding/output matrix once and includes both LayerNorms in every block and the final LayerNorm.

## Specification deviations

None. Only files in the section 2 allowlist were changed. No dependencies were added, no GPT reference implementation source was read, and no push was performed. The repository has no configured lint command or lint configuration; the specified pytest acceptance command passed.

## Implementation notes

- Pre-LN attention and MLP branches each feed a residual addition; final logits come from the final-normalized state.
- Output weights are tied directly to the token embedding transpose, so no duplicate vocabulary projection parameter exists.
- GPT initializes from one deterministic seeded stream because the reused L2 component constructors reset their local seed. Embedding scale is `0.02`; projection scale is `1/sqrt(d_model)`.
- The PyTorch Oracle matched logits, loss, and every named parameter gradient at `atol=rtol=1e-4`. The fixed-batch pre-test used only inline SGD in the test and showed a significant loss reduction in 50 steps.
