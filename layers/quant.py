"""Handwritten symmetric weight quantization and dequantize-before-matmul PTQ.

The public quant/dequant functions use [output, input] matrices, grouping along
input. INT8 stores signed bytes; INT4 packs two two's-complement nibbles per
byte, with the earlier input in the low nibble. Both formats store fp16 scales.
Project projections use [input, output], so QuantizedLinear transposes at the
boundary. No floating-point copy of a quantized weight is retained.
"""

import mlx.core as mx
import mlx.nn as nn

from .attention import CausalSelfAttention
from .transformer import MLP


def _group_size(bits: int, group_size: int | None, width: int) -> int:
    if type(bits) is not int or bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if group_size is None:
        group_size = width if bits == 8 else 64
    if type(group_size) is not int or group_size <= 0 or width % group_size:
        raise ValueError("group_size must be a positive divisor of input width")
    if bits == 4 and width % 2:
        raise ValueError("INT4 input width must be even for byte packing")
    return group_size


def quantize(w: mx.array, bits: int, group_size: int | None = None):
    """Return packed integers and fp16 scales for a finite [output, input] w.

    Each group starts from s = max(abs(w)) / (2**(bits-1)-1), rounded upward
    to fp16 so endpoints cannot saturate beyond the half-step error bound.
    Round-to-nearest-even uses this *stored* scale. Zero groups use s=1; tiny
    positive scales clamp to the smallest fp16 subnormal. We reject ranges
    that fp16 scales cannot represent.
    """
    if w.ndim != 2 or min(w.shape) <= 0:
        raise ValueError("w must be a non-empty [output, input] matrix")
    group_size = _group_size(bits, group_size, w.shape[1])
    if not mx.issubdtype(w.dtype, mx.floating):
        raise ValueError("w must have floating-point dtype")
    w = w.astype(mx.float32)
    if not bool(mx.all(mx.isfinite(w)).item()):
        raise ValueError("w must contain only finite values")
    grouped = w.reshape(w.shape[0], -1, group_size)
    qmax = 2 ** (bits - 1) - 1
    amax = mx.max(mx.abs(grouped), axis=-1)
    raw_scales = mx.where(amax == 0, 1.0, mx.maximum(amax / qmax, 2.0**-24))
    if bool(mx.any(raw_scales > 65504.0).item()):
        raise ValueError("weight range exceeds finite fp16 scales")
    scales = raw_scales.astype(mx.float16)
    # Positive fp16 bit patterns are ordered. Advance one ULP only when the
    # cast rounded down (especially important for subnormal scales).
    scale_bits = scales.view(mx.uint16)
    round_up = (scales.astype(mx.float32) < raw_scales).astype(mx.uint16)
    scales = (scale_bits + round_up).view(mx.float16)
    integers = mx.clip(
        mx.round(grouped / scales.astype(mx.float32)[..., None]), -qmax, qmax
    ).astype(mx.int8).reshape(w.shape)
    if bits == 8:
        q = integers
    else:
        nibbles = integers.astype(mx.uint8) & 15
        q = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    # Finish quantization before dropping source arrays or measuring inference.
    mx.eval(q, scales)
    return q, scales


def dequantize(
    q: mx.array, scales: mx.array, bits: int, group_size: int | None = None
) -> mx.array:
    """Unpack symmetric integers and reconstruct [output, input] in fp32."""
    if q.ndim != 2 or min(q.shape) <= 0:
        raise ValueError("q must be a non-empty rank-two packed matrix")
    width = q.shape[1] * (2 if bits == 4 else 1)
    group_size = _group_size(bits, group_size, width)
    expected_dtype = mx.int8 if bits == 8 else mx.uint8
    if q.dtype != expected_dtype:
        raise ValueError(f"q must have dtype {expected_dtype}")
    if scales.shape != (q.shape[0], width // group_size) or scales.dtype != mx.float16:
        raise ValueError("scales must be fp16 with shape [output, input // group_size]")
    if bits == 8:
        integers = q.astype(mx.float32)
    else:
        low = (q & 15).astype(mx.int32)
        high = (q >> 4).astype(mx.int32)
        nibbles = mx.stack((low, high), axis=-1).reshape(q.shape[0], width)
        integers = mx.where(nibbles >= 8, nibbles - 16, nibbles).astype(mx.float32)
    grouped = integers.reshape(q.shape[0], -1, group_size)
    return (grouped * scales.astype(mx.float32)[..., None]).reshape(q.shape[0], width)


class QuantizedLinear(nn.Module):
    """Bias-free x @ w, storing only packed q and scales for w.T."""

    def __init__(self, weight: mx.array, bits: int, group_size: int | None = None):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("weight must have shape [input, output]")
        self.bits = bits
        self.group_size = _group_size(bits, group_size, weight.shape[0])
        self.q, self.scales = quantize(weight.T, bits, self.group_size)

    def dequantized_weight(self) -> mx.array:
        """Reconstruct afresh on each forward; never cache fp32 weights."""
        return dequantize(self.q, self.scales, self.bits, self.group_size).T

    def __call__(self, x: mx.array) -> mx.array:
        return x.astype(mx.float32) @ self.dequantized_weight()


class _QuantizedAttention(CausalSelfAttention):
    """Reuse the L2/L4 forward verbatim through dequantizing weight views."""

    def __init__(self, attention, bits, group_size):
        nn.Module.__init__(self)
        for name in ("d_model", "n_head", "d_head", "rope_base", "rope"):
            setattr(self, name, getattr(attention, name))
        self.q_proj = QuantizedLinear(attention.wq, bits, group_size)
        self.k_proj = QuantizedLinear(attention.wk, bits, group_size)
        self.v_proj = QuantizedLinear(attention.wv, bits, group_size)
        self.o_proj = QuantizedLinear(attention.wo, bits, group_size)

    @property
    def wq(self):
        return self.q_proj.dequantized_weight()

    @property
    def wk(self):
        return self.k_proj.dequantized_weight()

    @property
    def wv(self):
        return self.v_proj.dequantized_weight()

    @property
    def wo(self):
        return self.o_proj.dequantized_weight()


class _QuantizedMLP(MLP):
    def __init__(self, mlp, bits, group_size):
        nn.Module.__init__(self)
        self.d_model = mlp.d_model
        self.up_proj = QuantizedLinear(mlp.w1, bits, group_size)
        self.down_proj = QuantizedLinear(mlp.w2, bits, group_size)

    @property
    def w1(self):
        return self.up_proj.dequantized_weight()

    @property
    def w2(self):
        return self.down_proj.dequantized_weight()


def quantize_model_(model, bits: int, group_size: int | None = None):
    """Replace only block projections in place; return the same GPT instance.

    INT8 defaults to one scale per output channel, INT4 to groups of 64 inputs.
    Embedding/tied head and all norms retain their original fp32 arrays. Build
    replacements before mutating the model, so invalid groups leave it intact.
    """
    if any(isinstance(block.attn, _QuantizedAttention) for block in model.blocks):
        raise ValueError("model is already quantized; reload fp32 before changing format")
    replacements = [
        (_QuantizedAttention(block.attn, bits, group_size),
         _QuantizedMLP(block.mlp, bits, group_size))
        for block in model.blocks
    ]
    for block, (attention, mlp) in zip(model.blocks, replacements):
        block.attn = attention
        block.mlp = mlp
    return model
