from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt, resample_poly
import soundfile as sf
import torch
import torch.nn.functional as F


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float, eps: float = 1e-12) -> float:
    return float(20.0 * math.log10(max(abs(x), eps)))


# -----------------------------
# Differentiable (training-time)
# -----------------------------

def _fir_lowpass(cutoff_hz: float, sr: int, numtaps: int = 101) -> torch.Tensor:
    """
    Windowed-sinc lowpass FIR. Returns 1D kernel [numtaps].
    """
    cutoff = float(cutoff_hz) / (sr / 2.0)
    cutoff = max(1e-4, min(0.999, cutoff))
    n = torch.arange(numtaps) - (numtaps - 1) / 2.0
    h = torch.where(
        n == 0,
        torch.tensor(2 * cutoff),
        torch.sin(2 * math.pi * cutoff * n) / (math.pi * n),
    )
    # Hann window
    w = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(numtaps) / (numtaps - 1))
    h = h * w
    h = h / h.sum().clamp_min(1e-8)
    return h


def _fir_highpass(cutoff_hz: float, sr: int, numtaps: int = 101) -> torch.Tensor:
    lp = _fir_lowpass(cutoff_hz, sr, numtaps=numtaps)
    hp = -lp
    hp[(numtaps - 1) // 2] += 1.0
    return hp


def _apply_fir(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    x: [B, 1, T]
    h: [K] FIR kernel
    """
    K = h.numel()
    h = h.to(device=x.device, dtype=x.dtype).view(1, 1, K)
    pad = (K - 1) // 2
    return F.conv1d(F.pad(x, (pad, pad), mode="reflect"), h)


def add_noise_snr_torch(x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """
    x: [B,1,T]
    snr_db: [B] desired SNR in dB (higher = less noise)
    """
    B = x.shape[0]
    noise = torch.randn_like(x)
    sig_pow = torch.mean(x ** 2, dim=(1, 2)).clamp_min(1e-8)  # [B]
    noise_pow = torch.mean(noise ** 2, dim=(1, 2)).clamp_min(1e-8)
    target_noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    scale = torch.sqrt(target_noise_pow / noise_pow).view(B, 1, 1)
    return torch.clamp(x + noise * scale, -1.0, 1.0)


def random_gain_torch(x: torch.Tensor, gain_db: torch.Tensor) -> torch.Tensor:
    g = (10.0 ** (gain_db / 20.0)).view(-1, 1, 1)
    return torch.clamp(x * g, -1.0, 1.0)


def speed_change_torch(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """
    factor > 1 => speed up (shorter); factor < 1 => slow down (longer)
    Keeps output length fixed by padding/trimming.
    """
    B, C, T = x.shape
    outs = []
    for b in range(B):
        f = float(factor[b].item())
        new_len = max(8, int(round(T / f)))
        y = F.interpolate(x[b:b+1], size=new_len, mode="linear", align_corners=False)
        if new_len < T:
            y = F.pad(y, (0, T - new_len))
        elif new_len > T:
            y = y[:, :, :T]
        outs.append(y)
    return torch.cat(outs, dim=0)


def random_crop_pad_torch(x: torch.Tensor, crop_len: torch.Tensor) -> torch.Tensor:
    """
    Removes crop_len[b] samples at a random position, then pads zeros to keep length.
    crop_len: [B] integer samples, 0..max
    """
    B, C, T = x.shape
    outs = []
    for b in range(B):
        c = int(crop_len[b].item())
        if c <= 0:
            outs.append(x[b:b+1])
            continue
        c = min(c, T - 1)
        start = torch.randint(0, T - c, (1,), device=x.device).item()
        y = torch.cat([x[b:b+1, :, :start], x[b:b+1, :, start + c :]], dim=-1)
        y = F.pad(y, (0, T - y.shape[-1]))
        outs.append(y)
    return torch.cat(outs, dim=0)


@dataclass
class TrainAttackConfig:
    noise_snr_db: Tuple[float, float] = (15.0, 40.0)
    gain_db: Tuple[float, float] = (-6.0, 6.0)
    resample_factor: Tuple[float, float] = (0.9, 1.1)
    lowpass_hz: Tuple[float, float] = (3500.0, 7000.0)
    highpass_hz: Tuple[float, float] = (20.0, 200.0)
    crop_seconds: Tuple[float, float] = (0.0, 0.2)

    p_lowpass: float = 0.5
    p_highpass: float = 0.3


def _speed_change_pair_torch(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    factor: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply the same speed change to x and (optionally) mask.

    x: [B,1,T]
    mask: [B,1,T] or None
    factor: [B]
    """
    B, C, T = x.shape
    outs_x = []
    outs_m = [] if mask is not None else None
    for b in range(B):
        f = float(factor[b].item())
        new_len = max(8, int(round(T / f)))
        xb = F.interpolate(x[b:b+1], size=new_len, mode="linear", align_corners=False)
        if new_len < T:
            xb = F.pad(xb, (0, T - new_len))
        elif new_len > T:
            xb = xb[:, :, :T]
        outs_x.append(xb)

        if mask is not None:
            mb = F.interpolate(mask[b:b+1], size=new_len, mode="linear", align_corners=False)
            if new_len < T:
                mb = F.pad(mb, (0, T - new_len))
            elif new_len > T:
                mb = mb[:, :, :T]
            outs_m.append(mb)

    x_out = torch.cat(outs_x, dim=0)
    if mask is None:
        return x_out, None
    m_out = torch.cat(outs_m, dim=0).clamp(0.0, 1.0)
    return x_out, m_out


def _random_crop_pad_pair_torch(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    crop_len: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply the same random crop+pad to x and (optionally) mask.

    crop_len: [B] int samples to remove.
    """
    B, C, T = x.shape
    outs_x = []
    outs_m = [] if mask is not None else None
    for b in range(B):
        c = int(crop_len[b].item())
        if c <= 0:
            outs_x.append(x[b:b+1])
            if mask is not None:
                outs_m.append(mask[b:b+1])
            continue
        c = min(c, T - 1)
        start = int(torch.randint(0, T - c, (1,), device=x.device).item())
        xb = torch.cat([x[b:b+1, :, :start], x[b:b+1, :, start + c :]], dim=-1)
        xb = F.pad(xb, (0, T - xb.shape[-1]))
        outs_x.append(xb)
        if mask is not None:
            mb = torch.cat([mask[b:b+1, :, :start], mask[b:b+1, :, start + c :]], dim=-1)
            mb = F.pad(mb, (0, T - mb.shape[-1]))
            outs_m.append(mb)

    x_out = torch.cat(outs_x, dim=0)
    if mask is None:
        return x_out, None
    m_out = torch.cat(outs_m, dim=0).clamp(0.0, 1.0)
    return x_out, m_out


class DifferentiableAttacks:
    def __init__(self, cfg: TrainAttackConfig, sample_rate: int):
        self.cfg = cfg
        self.sr = int(sample_rate)

    def __call__(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply training-time attacks.

        x: [B,1,T]
        mask: optional supervision mask [B,1,T] (1 where watermark should be present).

        Returns: (x_attacked, mask_attacked)
        """
        B, _, T = x.shape
        device = x.device

        # Noise (does not change time alignment)
        snr = torch.empty(B, device=device).uniform_(self.cfg.noise_snr_db[0], self.cfg.noise_snr_db[1])
        x = add_noise_snr_torch(x, snr)

        # Gain (does not change time alignment)
        g = torch.empty(B, device=device).uniform_(self.cfg.gain_db[0], self.cfg.gain_db[1])
        x = random_gain_torch(x, g)

        # Speed change (changes time alignment) -> apply to mask too
        f = torch.empty(B, device=device).uniform_(self.cfg.resample_factor[0], self.cfg.resample_factor[1])
        x, mask = _speed_change_pair_torch(x, mask, f)

        # Filters (optional; do NOT apply to mask)
        if torch.rand(()) < self.cfg.p_lowpass:
            cutoff = float(torch.empty(()).uniform_(self.cfg.lowpass_hz[0], self.cfg.lowpass_hz[1]).item())
            h = _fir_lowpass(cutoff, self.sr, numtaps=101)
            x = _apply_fir(x, h)
        if torch.rand(()) < self.cfg.p_highpass:
            cutoff = float(torch.empty(()).uniform_(self.cfg.highpass_hz[0], self.cfg.highpass_hz[1]).item())
            h = _fir_highpass(cutoff, self.sr, numtaps=101)
            x = _apply_fir(x, h)

        # Crop-pad (changes time alignment) -> apply to mask too
        max_crop = int(round(self.cfg.crop_seconds[1] * self.sr))
        if max_crop > 0:
            crop = torch.empty(B, device=device).uniform_(self.cfg.crop_seconds[0], self.cfg.crop_seconds[1])
            crop_len = torch.round(crop * self.sr).to(torch.int64)
            x, mask = _random_crop_pad_pair_torch(x, mask, crop_len)

        x = torch.clamp(x, -1.0, 1.0)
        if mask is not None:
            mask = mask.clamp(0.0, 1.0)
        return x, mask


# -----------------------
# Numpy/ffmpeg (eval-time)
# -----------------------

def add_noise_snr_np(x: np.ndarray, snr_db: float) -> np.ndarray:
    noise = np.random.randn(*x.shape).astype(np.float32)
    sig_pow = float(np.mean(x ** 2) + 1e-8)
    noise_pow = float(np.mean(noise ** 2) + 1e-8)
    target_noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    scale = math.sqrt(target_noise_pow / noise_pow)
    y = x + noise * scale
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def gain_np(x: np.ndarray, gain_db: float) -> np.ndarray:
    g = db_to_lin(gain_db)
    return np.clip(x * g, -1.0, 1.0).astype(np.float32)


def resample_speed_np(x: np.ndarray, factor: float, sr: int) -> np.ndarray:
    """
    factor >1 => speed up.
    Keeps length fixed by padding/trimming.
    """
    T = x.shape[0]
    new_len = max(8, int(round(T / factor)))
    # resample_poly wants integer ratios; approximate with gcd trick
    # Here we use direct polyphase via rational approximation with limited denom.
    # Simpler: use resample_poly with up/down computed from gcd if sr is fixed.
    # We'll do a generic resampling by interpreting 'factor' as ratio.
    # Use 1000 as common denom for decent precision.
    denom = 1000
    up = int(round(denom))
    down = int(round(denom * factor))
    y = resample_poly(x, up, down).astype(np.float32)

    if y.shape[0] < T:
        y = np.pad(y, (0, T - y.shape[0]), mode="constant")
    elif y.shape[0] > T:
        y = y[:T]
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def lowpass_np(x: np.ndarray, sr: int, cutoff_hz: float, order: int = 6) -> np.ndarray:
    nyq = 0.5 * sr
    wn = min(0.999, max(1e-4, cutoff_hz / nyq))
    sos = butter(order, wn, btype="lowpass", output="sos")
    y = sosfilt(sos, x).astype(np.float32)
    return np.clip(y, -1.0, 1.0)


def highpass_np(x: np.ndarray, sr: int, cutoff_hz: float, order: int = 6) -> np.ndarray:
    nyq = 0.5 * sr
    wn = min(0.999, max(1e-4, cutoff_hz / nyq))
    sos = butter(order, wn, btype="highpass", output="sos")
    y = sosfilt(sos, x).astype(np.float32)
    return np.clip(y, -1.0, 1.0)


def crop_pad_np(x: np.ndarray, crop_seconds: float, sr: int) -> np.ndarray:
    T = x.shape[0]
    c = int(round(crop_seconds * sr))
    if c <= 0:
        return x
    c = min(c, T - 1)
    start = np.random.randint(0, T - c)
    y = np.concatenate([x[:start], x[start + c :]], axis=0)
    if y.shape[0] < T:
        y = np.pad(y, (0, T - y.shape[0]), mode="constant")
    else:
        y = y[:T]
    return y.astype(np.float32)


def bitdepth_reduce_np(x: np.ndarray, bits: int) -> np.ndarray:
    bits = int(bits)
    levels = float(2 ** (bits - 1) - 1)  # signed
    y = np.round(x * levels) / levels
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _ffmpeg_codec_roundtrip(x: np.ndarray, sr: int, codec: str, bitrate_k: int) -> np.ndarray:
    """
    Encode/decode via ffmpeg using temporary files.
    codec: "mp3" or "aac"
    """
    codec = codec.lower()
    if codec not in ("mp3", "aac"):
        raise ValueError(f"Unsupported codec: {codec}")

    with tempfile.TemporaryDirectory() as td:
        in_wav = os.path.join(td, "in.wav")
        out_enc = os.path.join(td, f"enc.{codec}")
        out_wav = os.path.join(td, "out.wav")

        sf.write(in_wav, x, sr, subtype="PCM_16")

        # Encode
        # -vn no video; -y overwrite
        cmd_enc = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-vn",
            "-i", in_wav,
            "-b:a", f"{int(bitrate_k)}k",
            out_enc,
        ]
        subprocess.run(cmd_enc, check=True)

        # Decode back to wav
        cmd_dec = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-vn",
            "-i", out_enc,
            "-ar", str(sr),
            "-ac", "1",
            out_wav,
        ]
        subprocess.run(cmd_dec, check=True)

        y, _ = sf.read(out_wav, dtype="float32", always_2d=False)
        y = np.asarray(y).astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1).astype(np.float32)

    # match length
    if y.shape[0] < x.shape[0]:
        y = np.pad(y, (0, x.shape[0] - y.shape[0]), mode="constant")
    elif y.shape[0] > x.shape[0]:
        y = y[: x.shape[0]]
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def mp3_roundtrip_np(x: np.ndarray, sr: int, bitrate_k: int) -> np.ndarray:
    return _ffmpeg_codec_roundtrip(x, sr, codec="mp3", bitrate_k=bitrate_k)


def aac_roundtrip_np(x: np.ndarray, sr: int, bitrate_k: int) -> np.ndarray:
    return _ffmpeg_codec_roundtrip(x, sr, codec="aac", bitrate_k=bitrate_k)
