"""Validate the complete local model artifact contract before installing weights."""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from layers import BPETokenizer, GPT

MODEL_MANIFEST = Path(__file__).with_name("model-manifest.json")


def named_arrays(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from named_arrays(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from named_arrays(value, f"{prefix}.{index}")
    else:
        yield prefix, node


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_assets(checkpoint, tokenizer_path, model=None, manifest_path=None):
    checkpoint, tokenizer_path = Path(checkpoint), Path(tokenizer_path)
    for path in (checkpoint, tokenizer_path):
        if not path.is_file():
            raise ValueError(f"missing local artifact: {path}; see README asset preparation")
    tokenizer = BPETokenizer.load(str(tokenizer_path))
    model = model if model is not None else GPT()
    if tokenizer.vocab_size + 4 != model.vocab_size:
        raise ValueError("tokenizer ID space does not match model vocabulary")
    expected = dict(named_arrays(model.parameters()))
    # Keep the original model untouched until every tensor has been validated.
    with np.load(checkpoint, allow_pickle=False) as archive:
        if set(archive.files) != set(expected) or len(archive.files) != len(expected):
            raise ValueError("checkpoint parameter names do not match model")
        arrays = {}
        for name, template in expected.items():
            value = archive[name]
            if value.shape != template.shape or value.dtype != np.float32:
                raise ValueError(f"checkpoint shape/dtype mismatch: {name}")
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite checkpoint tensor: {name}")
            arrays[name] = mx.array(value)
    manifest = {
        "checkpoint_sha256": file_hash(checkpoint),
        "tokenizer_sha256": file_hash(tokenizer_path),
        "vocab_size": model.vocab_size,
        "parameters": sum(int(value.size) for value in arrays.values()),
    }
    if manifest_path is not None:
        expected_manifest = json.loads(Path(manifest_path).read_text())
        for name in ("checkpoint_sha256", "tokenizer_sha256", "vocab_size", "parameters"):
            if expected_manifest.get(name) != manifest[name]:
                raise ValueError(f"artifact manifest mismatch: {name}")
    for name, value in arrays.items():
        parts = name.split(".")
        current = model
        for part in parts[:-1]:
            current = current[int(part)] if part.isdigit() else getattr(current, part)
        setattr(current, parts[-1], value)
    mx.eval(model.parameters())
    return model, tokenizer, manifest


def register_training_artifacts(checkpoint, tokenizer_path, training_metrics, output):
    """Bind a locally trained checkpoint to its tokenizer without overwriting provenance."""
    output = Path(output)
    if output.exists():
        raise ValueError("manifest output already exists; choose a new path")
    model, _tokenizer, manifest = load_assets(checkpoint, tokenizer_path)
    metrics = json.loads(Path(training_metrics).read_text())
    for key in ("vocab_size", "n_layer", "d_model", "n_head", "ctx"):
        if metrics["model"].get(key) != getattr(model, key):
            raise ValueError(f"training metadata model mismatch: {key}")
    if metrics["corpus"].get("tokenizer_sha256") != manifest["tokenizer_sha256"]:
        raise ValueError("training metadata tokenizer mismatch")
    if metrics["model"].get("parameter_count") != manifest["parameters"]:
        raise ValueError("training metadata parameter count mismatch")
    manifest.update(training_config_hash=metrics["config_hash"],
                    training_metrics_sha256=file_hash(training_metrics))
    with output.open("x") as destination:
        destination.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register local FP32 GPT artifacts against training metadata")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--training-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    register_training_artifacts(args.checkpoint, args.tokenizer, args.training_metrics, args.output)
    print(f"Registered local artifacts in {args.output}")
