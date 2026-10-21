"""Mixed precision dtypes, fp32 Oracle, and the real 200-step L3 recipe."""

import mlx.core as mx
import numpy as np

from layers import GPT
from layers.optimizer import AdamW
from scripts.train import BF16TrainingEngine, TrainingEngine, flatten_arrays, run_bf16_training


def test_compute_gradients_bf16_loss_master_and_adam_fp32():
    model = GPT(vocab_size=17, n_layer=1, d_model=8, n_head=2, ctx=4, seed=19)
    optimizer = AdamW(weight_decay=0)
    original = np.asarray(model.token_embedding.weight).copy()
    engine = BF16TrainingEngine(model, optimizer)
    ids, targets = mx.array([[1, 2, 3, 4]]), mx.array([[2, 3, 4, 5]])
    loss, gradients = engine._loss_and_grad(engine.compute_model, ids, targets)
    mx.eval(loss, gradients)
    assert loss.dtype == mx.float32
    assert all(p.dtype == mx.bfloat16 for p in flatten_arrays(engine.compute_model.parameters()))
    assert all(g.dtype == mx.bfloat16 for g in flatten_arrays(gradients))
    np.testing.assert_array_equal(np.asarray(model.token_embedding.weight), original)
    fp_model = GPT(vocab_size=17, n_layer=1, d_model=8, n_head=2, ctx=4, seed=19)
    fp_engine = TrainingEngine(fp_model, AdamW(weight_decay=0))
    fp_loss, fp_gradients = fp_engine._loss_and_grad(fp_model, ids, targets)
    mx.eval(fp_loss, fp_gradients)
    assert abs(float(loss) - float(fp_loss)) / float(fp_loss) < 0.02
    bf = np.concatenate([np.asarray(g.astype(mx.float32)).ravel() for g in flatten_arrays(gradients)])
    fp = np.concatenate([np.asarray(g).ravel() for g in flatten_arrays(fp_gradients)])
    assert np.linalg.norm(bf - fp) / np.linalg.norm(fp) < 0.05
    engine.step(ids, targets, learning_rate=1e-5)
    assert all(p.dtype == mx.float32 for p in flatten_arrays(model.parameters()))
    assert all(p.dtype == mx.float32 for p in optimizer.first_moment.values())
    assert all(p.dtype == mx.float32 for p in optimizer.second_moment.values())
    after = np.asarray(model.token_embedding.weight)
    assert np.any(after != original)
    # Small master updates survive even when the compute copy rounds them away.
    rounded = np.asarray(model.token_embedding.weight.astype(mx.bfloat16).astype(mx.float32))
    assert np.any(after != rounded)


def test_bf16_200_steps_loss_finite_reduction_and_baseline_difference_recorded():
    record = run_bf16_training()
    training = record["training"]
    assert training["steps"] == 200
    assert training["master_weights_fp32"] and training["adam_states_fp32"]
    assert np.isfinite(training["step_train_loss_samples"]).all()
    assert training["heldout_loss_relative_reduction"] >= 0.28
    # The 2% prediction is recorded as evidence, deliberately not a quality gate.
    assert np.isfinite(record["comparison"]["final_loss_relative_difference"])
    assert "final_loss_difference_lt_2_percent" in record["prediction_verdicts"]
