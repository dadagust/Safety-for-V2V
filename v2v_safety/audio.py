from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

@dataclass
class AudioClip:
    wav: np.ndarray
    sr: int

def load_wav(path: str, target_sr: Optional[int] = None, mono: bool = True) -> AudioClip:
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim == 2 and mono:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    if target_sr is not None and sr != target_sr:
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
        wav = wav_t.squeeze(0).cpu().numpy().astype(np.float32)
        sr = target_sr

    peak = float(np.max(np.abs(wav)) + 1e-9)
    if peak > 1.0:
        wav = wav / peak
    return AudioClip(wav=wav, sr=sr)

def save_wav(path: str, wav: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, wav.astype(np.float32), sr)

def pad_or_crop(wav: np.ndarray, sr: int, seconds: float, random_crop: bool = True) -> np.ndarray:
    target_len = int(sr * seconds)
    if len(wav) == target_len:
        return wav
    if len(wav) < target_len:
        return np.pad(wav, (0, target_len - len(wav)), mode="constant")
    if not random_crop:
        return wav[:target_len]
    start = np.random.randint(0, len(wav) - target_len + 1)
    return wav[start:start + target_len]

def simple_vad_trim(wav: np.ndarray, sr: int, frame_ms: int = 30, thresh: float = 0.02) -> np.ndarray:
    # Lightweight energy-based VAD trim (fallback if webrtcvad isn't used).
    frame_len = int(sr * frame_ms / 1000.0)
    if frame_len <= 0:
        return wav
    n = len(wav)
    energies = []
    idxs = []
    for i in range(0, n - frame_len + 1, frame_len):
        frame = wav[i:i+frame_len]
        e = float(np.sqrt(np.mean(frame**2) + 1e-12))
        energies.append(e)
        idxs.append(i)
    if not energies:
        return wav
    energies = np.array(energies)
    mask = energies > thresh
    if not mask.any():
        return wav
    first = idxs[int(np.argmax(mask))]
    last_idx = int(len(mask) - 1 - np.argmax(mask[::-1]))
    last = idxs[last_idx] + frame_len
    return wav[first:last]

def mel_spectrogram_db(
    wav: torch.Tensor,
    sr: int,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> torch.Tensor:
    # wav: (B, T) -> (B, n_mels, frames) in dB
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0,
        center=True,
        pad_mode="reflect",
    )(wav)
    db = torchaudio.transforms.AmplitudeToDB(stype="power")(mel)
    return db
