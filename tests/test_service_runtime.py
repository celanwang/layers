"""Artifact contract checks use independent NumPy NPZ serialization."""
import numpy as np
import mlx.core as mx
import pytest

from layers import GPT, BPETokenizer
from server.runtime import load_assets, named_arrays
from server.app import load_engine, CHECKPOINT, TOKENIZER


def test_valid_assets_and_every_corruption_is_rejected_before_model_mutation(tmp_path):
    model = GPT(vocab_size=260, n_layer=1, d_model=8, n_head=2, ctx=16)
    params = {name: np.asarray(value) for name, value in named_arrays(model.parameters())}
    tokenizer_path = tmp_path / 'tokenizer.json'
    BPETokenizer().save(str(tokenizer_path))
    checkpoint = tmp_path / 'model.npz'
    np.savez(checkpoint, **params)
    loaded, tok, manifest = load_assets(checkpoint, tokenizer_path, model)
    assert loaded is model and tok.vocab_size + 4 == 260
    assert len(manifest['checkpoint_sha256']) == 64
    name = next(iter(params))
    for change in ('missing', 'extra', 'shape', 'dtype', 'nan'):
        bad = dict(params)
        if change == 'missing': del bad[name]
        if change == 'extra': bad['extra'] = np.ones(1, np.float32)
        if change == 'shape': bad[name] = np.zeros((1,), np.float32)
        if change == 'dtype': bad[name] = bad[name].astype(np.float16)
        if change == 'nan': bad[name] = np.full_like(bad[name], np.nan)
        np.savez(checkpoint, **bad)
        before = dict(named_arrays(model.parameters()))
        with pytest.raises(ValueError): load_assets(checkpoint, tokenizer_path, model)
        after = dict(named_arrays(model.parameters()))
        assert all(before[k] is after[k] for k in before)
    with pytest.raises(ValueError, match='ID space'):
        load_assets(checkpoint, tokenizer_path, GPT(vocab_size=261))
    with pytest.raises(ValueError, match='missing local artifact'):
        load_assets(tmp_path / 'missing', tokenizer_path)


def test_real_checkpoint_l6_l7_load_identical_and_match_decode():
    a = load_engine(CHECKPOINT, TOKENIZER, 'l6')
    b = load_engine(CHECKPOINT, TOKENIZER, 'l7')
    assert a.artifact_manifest == b.artifact_manifest
    from layers import Sampler
    for e in (a, b):
        e.add_request('ROMEO: ', Sampler(temperature=0), 8)
    assert [event.token_id for event in a.run()] == [event.token_id for event in b.run()]
    assert a.pool.allocated_blocks == b.pool.allocated_blocks == 0
    with pytest.raises(ValueError, match='engine must'):
        load_engine(engine_name='unknown')


def test_manifest_binds_tokenizer_and_checkpoint_before_install(tmp_path):
    import json
    from server.runtime import file_hash
    model = GPT(vocab_size=260, n_layer=1, d_model=8, n_head=2, ctx=16)
    params = {name: np.asarray(value) for name, value in named_arrays(model.parameters())}
    checkpoint, tokenizer = tmp_path / 'model.npz', tmp_path / 'tokenizer.json'
    np.savez(checkpoint, **params)
    BPETokenizer().save(str(tokenizer))
    manifest = tmp_path / 'manifest.json'
    expected = {'checkpoint_sha256':file_hash(checkpoint), 'tokenizer_sha256':file_hash(tokenizer),
                'vocab_size':260, 'parameters':sum(v.size for v in params.values())}
    manifest.write_text(json.dumps(expected))
    assert load_assets(checkpoint, tokenizer, model, manifest)[2] == expected
    before = dict(named_arrays(model.parameters()))
    tokenizer.write_text(tokenizer.read_text() + '\n')
    with pytest.raises(ValueError, match='tokenizer_sha256'):
        load_assets(checkpoint, tokenizer, model, manifest)
    assert all(value is dict(named_arrays(model.parameters()))[name] for name, value in before.items())


def test_register_reproduction_requires_matching_training_provenance(tmp_path):
    import json
    from server.runtime import register_training_artifacts, MODEL_MANIFEST
    metadata = tmp_path / 'training.json'
    baseline = json.loads(__import__('pathlib').Path('results/l3/train-metrics-500.json').read_text())
    metadata.write_text(json.dumps(baseline))
    output = tmp_path / 'local.json'
    manifest = register_training_artifacts(CHECKPOINT, TOKENIZER, metadata, output)
    assert manifest['training_config_hash'] == baseline['config_hash']
    assert output.exists()
    with pytest.raises(ValueError, match='already exists'):
        register_training_artifacts(CHECKPOINT, TOKENIZER, metadata, output)
    baseline['corpus']['tokenizer_sha256'] = 'wrong'
    metadata.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match='tokenizer mismatch'):
        register_training_artifacts(CHECKPOINT, TOKENIZER, metadata, tmp_path / 'bad.json')
