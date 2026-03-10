from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch

from .model import Encoder1D


def embed_random_window(
    x: torch.Tensor,
    bits: torch.Tensor,
    encoder: Encoder1D,
    watermark_len: int,
    p_embed: float,
    start_mode: str = "random",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Embed watermark into a single contiguous window for a subset of samples.

    Args:
        x: [B,1,T]
        bits: [B,nbits]
        encoder: watermark encoder
        watermark_len: window length in samples
        p_embed: probability that a given sample is watermarked
        start_mode: "random" or "prefix" (start at 0)

    Returns:
        y: [B,1,T]
        mask: [B,1,T] 1 where watermark is present
    """
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f"Expected x [B,1,T], got {x.shape}")
    if bits.ndim != 2:
        raise ValueError(f"Expected bits [B,nbits], got {bits.shape}")

    B, _, T = x.shape
    device = x.device
    y = x.clone()
    mask = torch.zeros((B, 1, T), device=device, dtype=x.dtype)

    for b in range(B):
        if torch.rand((), device=device).item() > float(p_embed):
            continue

        if watermark_len >= T:
            start = 0
            wlen = T
        else:
            if str(start_mode).lower() == "prefix":
                start = 0
            else:
                start = int(torch.randint(0, T - watermark_len + 1, (1,), device=device).item())
            wlen = watermark_len

        seg = x[b:b+1, :, start : start + wlen]
        y_seg, _residual = encoder(seg, bits[b:b+1])
        y[b:b+1, :, start : start + wlen] = y_seg
        mask[b:b+1, :, start : start + wlen] = 1.0

    return y, mask


def embed_full(
    x: torch.Tensor,
    bits: torch.Tensor,
    encoder: Encoder1D,
    do_embed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Embed watermark across the entire segment for samples where do_embed[b]=True."""
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f"Expected x [B,1,T], got {x.shape}")
    B, _, T = x.shape
    y = x.clone()
    mask = torch.zeros((B, 1, T), device=x.device, dtype=x.dtype)
    for b in range(B):
        if not bool(do_embed[b].item()):
            continue
        yb, _ = encoder(x[b:b+1], bits[b:b+1])
        y[b:b+1] = yb
        mask[b:b+1] = 1.0
    return y, mask


def _carve_hole_1d(mask_1d: torch.Tensor, start: int, end: int, ramp: int, protect_prefix: int) -> None:
    """In-place: set mask[start:end]=0 with optional linear ramps at boundaries.

    mask_1d: [T]
    start/end: indices
    ramp: samples
    protect_prefix: first protect_prefix samples should remain 1.0
    """
    T = int(mask_1d.numel())
    start = max(int(start), 0)
    end = min(int(end), T)
    if end <= start:
        return

    # hard zero in the hole
    mask_1d[start:end] = 0.0

    if ramp <= 0:
        # re-enforce protected prefix
        if protect_prefix > 0:
            mask_1d[:protect_prefix] = 1.0
        return

    # ramp down (left edge)
    left0 = max(protect_prefix, start - ramp)
    left_len = start - left0
    if left_len > 1:
        mask_1d[left0:start] = torch.linspace(1.0, 0.0, left_len, device=mask_1d.device, dtype=mask_1d.dtype)
    elif left_len == 1:
        mask_1d[left0:start] = 0.0

    # ramp up (right edge)
    right1 = min(T, end + ramp)
    right_len = right1 - end
    if right_len > 1:
        mask_1d[end:right1] = torch.linspace(0.0, 1.0, right_len, device=mask_1d.device, dtype=mask_1d.dtype)
    elif right_len == 1:
        mask_1d[end:right1] = 0.0

    # re-enforce protected prefix
    if protect_prefix > 0:
        mask_1d[:protect_prefix] = 1.0


def embed_dropout_holes(
    x: torch.Tensor,
    bits: torch.Tensor,
    encoder: Encoder1D,
    sample_rate: int,
    p_embed: float,
    ensure_prefix_seconds: float = 0.5,
    num_holes: Tuple[int, int] = (0, 3),
    hole_seconds: Tuple[float, float] = (0.25, 1.0),
    ramp_ms: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Variant A-style embedding: watermark is present almost everywhere, but with random "holes".

    This trains a detector to *localize* watermark presence and also makes it possible to
    detect quickly from the beginning (low latency) by keeping the prefix protected.

    Args:
        x: [B,1,T]
        bits: [B,nbits]
        encoder: watermark encoder
        sample_rate: Hz
        p_embed: probability sample is watermarked at all
        ensure_prefix_seconds: first N seconds are guaranteed to be watermarked (if embedded)
        num_holes: (min,max) number of unwatermarked holes per sample
        hole_seconds: (min,max) hole duration in seconds
        ramp_ms: crossfade ramp length around holes

    Returns:
        y: watermarked audio [B,1,T]
        mask: watermark presence mask [B,1,T]
    """
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f"Expected x [B,1,T], got {x.shape}")

    B, _, T = x.shape
    device = x.device
    y = x.clone()
    mask = torch.zeros((B, 1, T), device=device, dtype=x.dtype)

    protect = int(round(float(ensure_prefix_seconds) * int(sample_rate)))
    protect = max(0, min(protect, T))

    min_holes, max_holes = int(num_holes[0]), int(num_holes[1])
    min_holes = max(0, min_holes)
    max_holes = max(min_holes, max_holes)

    min_h = int(round(float(hole_seconds[0]) * int(sample_rate)))
    max_h = int(round(float(hole_seconds[1]) * int(sample_rate)))
    min_h = max(1, min_h)
    max_h = max(min_h, max_h)

    ramp = int(round(float(ramp_ms) / 1000.0 * int(sample_rate)))
    ramp = max(0, ramp)

    for b in range(B):
        if torch.rand((), device=device).item() > float(p_embed):
            continue

        # get residual from encoder
        y_full, residual = encoder(x[b:b+1], bits[b:b+1])
        # Start with fully watermarked
        m = torch.ones((T,), device=device, dtype=x.dtype)

        # Carve random holes after the protected prefix
        nh = int(torch.randint(min_holes, max_holes + 1, (1,), device=device).item())
        for _ in range(nh):
            # if we cannot fit a hole, skip
            if protect + min_h + 1 >= T:
                break
            hlen = int(torch.randint(min_h, max_h + 1, (1,), device=device).item())
            hlen = min(hlen, max(1, T - protect - 1))
            start = int(torch.randint(protect, T - hlen, (1,), device=device).item())
            end = start + hlen
            _carve_hole_1d(m, start, end, ramp=ramp, protect_prefix=protect)

        # ensure prefix is watermarked
        if protect > 0:
            m[:protect] = 1.0

        m3 = m.view(1, 1, T)
        y_b = torch.clamp(x[b:b+1] + residual * m3, -1.0, 1.0)
        y[b:b+1] = y_b
        mask[b:b+1] = m3

    return y, mask
