from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    """
    x: (..., T)
    mask: (..., T) broadcastable to x
    Returns mean over `dim` with mask weights.
    """
    masked = x * mask
    denom = mask.sum(dim=dim, keepdim=False).clamp_min(eps)
    return masked.sum(dim=dim, keepdim=False) / denom


def stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    """
    x: [B, T] float
    returns |STFT|: [B, F, TT]
    """
    window = torch.hann_window(win, device=x.device, dtype=x.dtype)
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=True,
        return_complex=True,
    )
    return torch.abs(X)


def multiscale_stft_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    scales: List[Dict[str, int]],
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Multi-scale spectral L1 loss on magnitude spectrograms.
    x,y: [B, 1, T]
    """
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expected [B, 1, T]. Got x={x.shape}, y={y.shape}")
    xb = x.squeeze(1)
    yb = y.squeeze(1)

    losses = []
    for s in scales:
        n_fft = int(s["n_fft"])
        hop = int(s["hop"])
        win = int(s.get("win", n_fft))
        X = stft_mag(xb, n_fft=n_fft, hop=hop, win=win)
        Y = stft_mag(yb, n_fft=n_fft, hop=hop, win=win)

        # Normalize by magnitude scale to reduce sensitivity to overall loudness
        denom = (Y.mean(dim=(1, 2), keepdim=True) + eps)
        losses.append(torch.mean(torch.abs(X - Y) / denom))
    return torch.stack(losses).mean()


def mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((x - y) ** 2)


def snr_db(clean: torch.Tensor, test: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Returns SNR in dB, batchwise.
    clean,test: [B, 1, T]
    """
    n = test - clean
    sp = torch.mean(clean ** 2, dim=(1, 2))
    np = torch.mean(n ** 2, dim=(1, 2)).clamp_min(eps)
    return 10.0 * torch.log10(sp / np)


def l1_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - y))


def snr_hinge_loss(
    clean: torch.Tensor,
    test: torch.Tensor,
    target_db: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize outputs whose SNR falls below the target."""
    snr = snr_db(clean, test, eps=eps)
    target = torch.as_tensor(float(target_db), device=clean.device, dtype=clean.dtype)
    return torch.relu(target - snr).mean()
