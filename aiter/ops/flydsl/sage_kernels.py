# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Sage Attention APIs (CDNA / gfx942 / gfx950).

Wraps the FlyDSL ``sage_attn_cdna`` kernel with:
  - Build cache keyed by (num_q_heads, num_kv_heads, head_dim, causal, waves_per_eu).
  - Automatic Q/K Int8 + V FP8 quantization via ``sage_quant``.
  - BSHD ([B, S, H, D]) input/output convention.
  - Seq_len padding to the kernel's tile size (multiple of BLOCK_M).

The kernel implements self-attention and GQA. Cross-attention (different Q
and KV seq_len) is also supported since Q and KV shapes may differ.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Optional

import torch

from aiter.utility.dtypes import fp8 as _fp8_dtype
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    sage_quant as _triton_sage_quant,
)
from .sage_quant import flydsl_sage_quant as _flydsl_sage_quant

from .kernels.sage_attn_cdna import build_sage_attn_cdna_module


def _sage_quant(*args, **kwargs):
    """Dispatch to FlyDSL or Triton sage_quant per FLYDSL_SAGE_QUANT_BACKEND env var.

    Default: ``"flydsl"`` — single fused Q INT8 + K INT8 + V FP8 launch
    (~1.18-1.20x Triton on small/medium shapes, parity on long-S).
    Set ``FLYDSL_SAGE_QUANT_BACKEND=triton`` to fall back to the per-op
    Triton wrappers.
    """
    backend = os.environ.get("FLYDSL_SAGE_QUANT_BACKEND", "flydsl").lower()
    if backend == "flydsl":
        return _flydsl_sage_quant(*args, **kwargs)
    v_transposed = kwargs.pop("v_transposed", False)
    result = _triton_sage_quant(*args, **kwargs)
    if not v_transposed:
        return result
    v_fp8 = result[4]
    block_k = int(kwargs.get("BLKK", 64))
    seq_k = v_fp8.shape[1]
    seq_k_pad = ((seq_k + block_k - 1) // block_k) * block_k
    if seq_k_pad != seq_k:
        v_fp8 = torch.nn.functional.pad(v_fp8, (0, 0, 0, 0, 0, seq_k_pad - seq_k))
    batch, _, heads, dim = v_fp8.shape
    v_fp8 = (
        v_fp8.permute(0, 2, 3, 1)
        .reshape(batch, heads, dim, seq_k_pad // block_k, block_k)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    return result[:4] + (v_fp8,) + result[5:]


def _estimate_w256_lds_bytes(
    block_n: int,
    head_dim: int,
    *,
    pipe_stages: int = 2,
    dma_k: bool = False,
) -> int:
    """Mirror w256 sage_attn_cdna LDS sizing (gfx942 column-major V)."""
    k_stride = head_dim if dma_k else head_dim + 16
    v_stride = block_n + 16
    lds_k = block_n * k_stride
    lds_v = head_dim * v_stride
    return (lds_k + lds_v) * pipe_stages


def _estimate_legacy_lds_bytes(
    block_n: int,
    head_dim: int,
    *,
    pipe_stages: int = 2,
    use_fp8_pv: bool = False,
    dma_k: bool = False,
) -> int:
    """Mirror legacy512 sage_attn_cdna_gfx942 LDS sizing."""
    k_stride = head_dim if dma_k else head_dim + 8
    if use_fp8_pv:
        lds_v = head_dim * (block_n + 16)
    else:
        lds_v = block_n * (head_dim + 4) * 2
    lds_k = block_n * k_stride
    return (lds_k + lds_v) * pipe_stages


def _gfx942_pipe_stages_for_tile(
    block_n: int,
    head_dim: int,
    *,
    use_w256: bool,
    use_fp8_pv: bool,
) -> int:
    """Return 1 or 2 pipe stages; auto-fallback when double-buffer exceeds 64 KiB."""
    _pipe_env = os.environ.get("SAGE_PIPE_STAGES", "2")
    if _pipe_env in ("0", "1"):
        return 1
    est_fn = _estimate_w256_lds_bytes if use_w256 else _estimate_legacy_lds_bytes
    kwargs = {"pipe_stages": 2}
    if not use_w256:
        kwargs["use_fp8_pv"] = use_fp8_pv
    if est_fn(block_n, head_dim, **kwargs) <= 65536:
        return 2
    return 1


def _hybrid_triton_attn(
    q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale,
    *, softmax_scale, causal, block_m, block_n, layout,
):
    """FlyDSL quant tensors + Triton A3 attn (B2 / production fast path)."""
    from aiter.ops.triton.attention.fav3_sage import fav3_sage_func

    a3_cfg = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "waves_per_eu": 2,
        "PRE_LOAD_V": False,
        "num_stages": 2,
        "num_warps": 4,
    }
    out = fav3_sage_func(
        q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=False,
        layout=layout,
        config=a3_cfg,
    )
    return out[0] if isinstance(out, tuple) else out


# keep the legacy name `sage_quant` working below
sage_quant = _sage_quant

__all__ = [
    "flydsl_sage_attn_func",
]

# Tile size baked into the CDNA kernel. Seq_len_q must be a multiple of this.
_KERNEL_BLOCK_M = 256


def _check_cdna_arch(device: torch.device) -> str:
    """Return arch string; raise if not gfx942/gfx950."""
    try:
        arch = torch.cuda.get_device_properties(device.index).gcnArchName
    except Exception:
        arch = ""
    arch_base = arch.lower().split(":")[0] if arch else ""
    if not (arch_base.startswith("gfx942") or arch_base.startswith("gfx950")):
        raise ValueError(
            f"flydsl_sage_attn_func requires gfx942 or gfx950, got {arch!r}"
        )
    return arch_base


@lru_cache(maxsize=128)
def _get_kernel(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    waves_per_eu: int,
    block_m: int,
    block_n: int,
    use_bias: bool = False,
    use_fp8_pv: bool = False,
    split_k: int = 1,
    gfx942_w256: bool = True,
    v_transposed: bool = False,
    bias_block_m: Optional[int] = None,
):
    return build_sage_attn_cdna_module(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        waves_per_eu=waves_per_eu,
        block_m=block_m,
        block_n=block_n,
        use_bias=use_bias,
        use_fp8_pv=use_fp8_pv,
        split_k=split_k,
        gfx942_w256=gfx942_w256,
        v_transposed=v_transposed,
        bias_block_m=bias_block_m,
    )


def _pick_kv_split_k(
    batch: int,
    num_q_heads: int,
    seq_q: int,
    seq_k: int,
    block_m: int,
    block_n: int,
    cu_count: int,
) -> int:
    """Pick KV split count for low-occupancy shapes (gemm-playground Split-K)."""
    env = os.environ.get("SAGE_KV_SPLIT_K", "0")
    if env in ("0", ""):
        return 1
    if env == "1":
        return 1
    if env == "auto":
        grid = batch * num_q_heads * ((seq_q + block_m - 1) // block_m)
        if seq_q > 2048 or seq_k > 2048:
            return 1
        if grid >= cu_count:
            return 1
        num_kv_tiles = (seq_k + block_n - 1) // block_n
        if num_kv_tiles >= 2:
            return 2
    else:
        return max(1, int(env))
    return 1


def flydsl_sage_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    layout: str = "bshd",
    waves_per_eu: int = 2,
    block_m: Optional[int] = None,
    block_n: int = 128,
    stream: Optional[torch.cuda.Stream] = None,
    q_smooth: bool = False,
) -> torch.Tensor:
    """Run FlyDSL Sage Attention on CDNA (gfx942/MI300X, gfx950/MI350).

    Internally quantizes Q/K to Int8 and V to FP8, then runs the FlyDSL
    flash-attention kernel with Int8 MFMA (GEMM1) and BF16 MFMA (GEMM2).

    Args:
        q, k, v: tensors with shape ``[batch, seq_len, num_heads, head_dim]``
            (BSHD layout, default) or ``[batch, num_heads, seq_len, head_dim]``
            (BHSD layout, set ``layout="bhsd"``).
            Supported dtypes: bf16, fp16, fp32.
            Must reside on a CUDA/HIP device.
        softmax_scale: scaling factor for QK^T (default: 1/sqrt(head_dim)).
        causal: apply causal masking when ``True``.
        layout: ``"bshd"`` (default) or ``"bhsd"``.
        waves_per_eu: kernel occupancy hint.
        block_m: Q tile size. If ``None`` (default), auto-selected per shape:
            BLOCK_M=128 when grid_at_BM256 < CU count (small-S/low-occupancy
            shapes win ~17% from the doubled grid), else BLOCK_M=256.
            On gfx942, BLOCK_M=256 with BLOCK_N=128 requires single-buffer
            LDS (``SAGE_PIPE_STAGES=1`` or auto-fallback).
        block_n: KV tile size (default 128).
        stream: optional CUDA/HIP stream. Defaults to current stream for q.device.

    Returns:
        Output tensor with the same BSHD shape as ``q``, dtype BF16.

    Raises:
        ValueError: if shapes/dtypes/devices are incompatible or the GPU is
            not gfx942/gfx950.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_sage_attn_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(
            f"q/k/v must be on the same device: q={q.device} k={k.device} v={v.device}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtype must match: {q.dtype}/{k.dtype}/{v.dtype}")
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if q.dtype not in supported_dtypes:
        raise ValueError(
            f"flydsl_sage_attn_func supports fp16/bf16/fp32 inputs, got {q.dtype}"
        )
    if q.dim() != 4:
        raise ValueError(f"expected 4D tensor, got rank {q.dim()} ({tuple(q.shape)})")
    if layout not in ("bshd", "bhsd"):
        raise ValueError(f"layout must be 'bshd' or 'bhsd', got {layout!r}")

    arch_base = _check_cdna_arch(q.device)
    is_gfx942 = arch_base.startswith("gfx942")

    if layout == "bshd":
        batch, seq_q, num_q_heads, head_dim = q.shape
        _, seq_k, num_kv_heads, _ = k.shape
    else:
        batch, num_q_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    if head_dim < 64 or head_dim % 32 != 0:
        raise ValueError(
            f"kernel requires head_dim >= 64 and head_dim % 32 == 0, got {head_dim}"
        )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    try:
        cu_count = torch.cuda.get_device_properties(
            q.device.index
        ).multi_processor_count
    except Exception:
        cu_count = 256

    # Auto-pick BLOCK_M for occupancy. BM=256 only saturates the GPU once
    # batch * num_q_heads * ceil(seq_q/256) >= CU count. For small-S/H1
    # shapes the BM=256 grid is well below CU count and BM=128 (which doubles
    # the grid) wins +17%; for shapes that already saturate, BM=256 wins
    # (larger MFMA chains, single coop-load batch).
    if block_m is None:
        grid_at_bm256 = batch * num_q_heads * ((seq_q + 255) // 256)
        block_m = 128 if grid_at_bm256 < cu_count else 256
    if is_gfx942:
        attn_block_m = block_m
        # The experimental BM64 path is opt-in only. Auto-selection produced
        # incorrect qwen1024 output with the gfx942 blocked-V kernel; BM256 is
        # both correct and equally fast for that production shape.
        _small_m_env = os.environ.get("SAGE_SMALL_M", "0")
        if seq_q <= 2048 and _small_m_env in ("1", "force"):
            attn_block_m = min(block_m, 64)
        quant_blk_q = attn_block_m
        _bn_env = os.environ.get("FLYDSL_GFX942_BLOCK_N", "")
        if _bn_env:
            block_n = int(_bn_env)
        elif block_n == 128:
            # gfx942 w256: 256x64 fits double-buffered K/V LDS and is the
            # measured best tile for both Qwen and production long sequences.
            block_n = 64
    else:
        attn_block_m = block_m
        quant_blk_q = block_m

    fp8_dtype = _fp8_dtype
    fp8_max = torch.finfo(fp8_dtype).max

    use_fp8_pv = (
        is_gfx942
        and os.environ.get("FLYDSL_GFX942_FP8_PV", "0").lower() not in ("0", "")
    )
    attn_backend = os.environ.get("FLYDSL_SAGE_ATTN_BACKEND", "flydsl").lower()

    smoothq_block_m = 128 if q_smooth else quant_blk_q
    # Keep ordinary Sage on its original six-result launch graph. SmoothQ is
    # opt-in and receives a separately compiled bias specialization.
    quant_result = sage_quant(
        q,
        k,
        v,
        FP8_TYPE=fp8_dtype,
        FP8_MAX=fp8_max,
        BLKQ=smoothq_block_m,
        BLKK=block_n,
        sm_scale=softmax_scale,
        layout=layout,
        v_transposed=is_gfx942 and layout == "bshd",
        q_smoothing=q_smooth,
    )
    q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale = quant_result[:6]
    delta_s = quant_result[6] if q_smooth else None

    # Pad seq_q to multiple of attn_block_m so the kernel tile loop is clean
    seq_q_pad = ((seq_q + attn_block_m - 1) // attn_block_m) * attn_block_m
    n_pad_q = seq_q_pad - seq_q

    # Contiguous BSHD views for kernel (convert BHSD if needed)
    if layout == "bhsd":
        q_int8 = q_int8.permute(0, 2, 1, 3).contiguous()
        k_int8 = k_int8.permute(0, 2, 1, 3).contiguous()
        v_fp8 = v_fp8.permute(0, 2, 1, 3).contiguous()
        # q_scale / k_scale are [batch, heads, blocks] — no permute needed
        # v_scale is [batch, heads, ...] — no permute needed
    else:
        q_int8 = q_int8.contiguous()
        k_int8 = k_int8.contiguous()
        v_fp8 = v_fp8.contiguous()

    if n_pad_q > 0:
        # Pad Q seq dim (dim=1 in BSHD)
        q_int8 = torch.nn.functional.pad(q_int8, (0, 0, 0, 0, 0, n_pad_q))

    if attn_backend in ("triton", "hybrid"):
        out_h = _hybrid_triton_attn(
            q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale,
            softmax_scale=softmax_scale,
            causal=causal,
            block_m=attn_block_m,
            block_n=block_n,
            layout=layout,
        )
        if n_pad_q > 0:
            out_h = out_h[:, :seq_q, :, :].contiguous()
        if layout == "bhsd":
            out_h = out_h.permute(0, 2, 1, 3).contiguous()
        return out_h

    # Allocate BF16 output in BSHD (padded)
    o_shape = (batch, seq_q_pad, num_q_heads, head_dim)
    o = torch.empty(o_shape, dtype=torch.bfloat16, device=q.device)
    num_q_blocks = q_scale.shape[2]
    # SmoothQ currently targets the production dense path. Split-K remains
    # disabled until its state merge is available in this branch.
    split_k = 1
    _sk_dummy = torch.empty(1, device=q.device, dtype=torch.float32)

    with torch.cuda.device(q.device.index):
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        if launch_stream.device != q.device:
            raise ValueError(
                f"`stream` must be on {q.device}, got {launch_stream.device}"
            )

        _w256_env = os.environ.get("SAGE_GFX942_W256", "auto")
        if _w256_env == "auto":
            _gfx942_w256 = True
        elif _w256_env in ("compact", "c256"):
            _gfx942_w256 = False
        else:
            _gfx942_w256 = is_gfx942 and _w256_env not in ("0", "")
        exe = _get_kernel(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            causal=causal,
            waves_per_eu=waves_per_eu,
            block_m=attn_block_m,
            block_n=block_n,
            use_bias=q_smooth,
            use_fp8_pv=use_fp8_pv,
            split_k=split_k,
            gfx942_w256=_gfx942_w256,
            v_transposed=is_gfx942 and layout == "bshd",
            bias_block_m=smoothq_block_m,
        )
        bias_arg = (
            delta_s.contiguous()
            if q_smooth
            else torch.empty(1, device=q.device, dtype=torch.float32)
        )

        exe(
            q_int8.reshape(-1),
            k_int8.reshape(-1),
            v_fp8.reshape(-1),
            o.reshape(-1),
            q_scale,
            k_scale,
            v_scale,
            bias_arg,
            _sk_dummy,
            _sk_dummy,
            _sk_dummy,
            batch,
            seq_q_pad,
            seq_k,
            num_q_blocks,
            stream=launch_stream,
        )

    if n_pad_q > 0:
        o = o[:, :seq_q, :, :].contiguous()

    # Convert back to BHSD if caller requested it
    if layout == "bhsd":
        o = o.permute(0, 2, 1, 3).contiguous()

    return o
