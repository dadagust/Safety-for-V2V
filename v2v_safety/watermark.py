from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.signal import stft, istft, get_window

@dataclass
class WatermarkConfig:
    seed: int = 1337
    delta_db: float = 0.30
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    window: str = "hann"
    sr: int = 16000
    f_min_hz: float = 500.0
    f_max_hz: float = 7000.0
    threshold: float = 0.12

def _pn_sequence(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n,), dtype=np.int32)
    return (bits * 2 - 1).astype(np.float32)  # {-1, +1}

def embed_zero_bit_spread_spectrum(wav: np.ndarray, cfg: WatermarkConfig) -> np.ndarray:
    x = wav.astype(np.float32)

    win = get_window(cfg.window, cfg.win_length, fftbins=True)
    f, _, Zxx = stft(
        x, fs=cfg.sr, window=win,
        nperseg=cfg.win_length, noverlap=cfg.win_length - cfg.hop_length,
        nfft=cfg.n_fft, boundary=None, padded=False, return_onesided=True
    )
    mag = np.abs(Zxx) + 1e-12
    phase = np.angle(Zxx)

    fmask = (f >= cfg.f_min_hz) & (f <= cfg.f_max_hz)
    idx = np.where(fmask)[0]
    if idx.size == 0:
        return x

    pn = _pn_sequence(idx.size, cfg.seed)[:, None]  # (bins,1)
    mag_db = 20.0 * np.log10(mag)
    mag_db[idx, :] = mag_db[idx, :] + cfg.delta_db * pn

    mag2 = 10.0 ** (mag_db / 20.0)
    Z2 = mag2 * np.exp(1j * phase)
    _, xw = istft(
        Z2, fs=cfg.sr, window=win,
        nperseg=cfg.win_length, noverlap=cfg.win_length - cfg.hop_length,
        nfft=cfg.n_fft, input_onesided=True, boundary=None
    )
    xw = xw.astype(np.float32)

    if len(xw) > len(x):
        xw = xw[:len(x)]
    elif len(xw) < len(x):
        xw = np.pad(xw, (0, len(x) - len(xw)))

    peak = float(np.max(np.abs(xw)) + 1e-9)
    if peak > 0.999:
        xw = xw / peak * 0.999
    return xw

def detect_zero_bit_spread_spectrum(wav: np.ndarray, cfg: WatermarkConfig) -> Tuple[bool, float]:
    x = wav.astype(np.float32)
    win = get_window(cfg.window, cfg.win_length, fftbins=True)
    f, _, Zxx = stft(
        x, fs=cfg.sr, window=win,
        nperseg=cfg.win_length, noverlap=cfg.win_length - cfg.hop_length,
        nfft=cfg.n_fft, boundary=None, padded=False, return_onesided=True
    )
    mag = np.abs(Zxx) + 1e-12
    mag_db = 20.0 * np.log10(mag)

    fmask = (f >= cfg.f_min_hz) & (f <= cfg.f_max_hz)
    idx = np.where(fmask)[0]
    if idx.size == 0:
        return False, 0.0

    pn = _pn_sequence(idx.size, cfg.seed).astype(np.float32)
    band = mag_db[idx, :]
    band = band - band.mean(axis=0, keepdims=True)

    dots = (band.T @ pn) / (np.linalg.norm(pn) + 1e-9)
    denom = np.linalg.norm(band, axis=0) + 1e-9
    corr = dots / denom
    score = float(np.mean(corr))
    return (score >= cfg.threshold), score
