from __future__ import annotations
import os, random, subprocess, tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio

from v2v_safety.audio import load_wav, pad_or_crop, simple_vad_trim, mel_spectrogram_db

@dataclass
class AugmentConfig:
    enable: bool = True
    noise_prob: float = 0.5
    noise_snr_db_min: float = 10.0
    noise_snr_db_max: float = 30.0
    reverb_prob: float = 0.2
    rir_dir: Optional[str] = None
    codec_prob: float = 0.0
    codec: str = "opus"
    codec_bitrate: int = 16000

def _add_noise(wav: torch.Tensor, snr_db: float) -> torch.Tensor:
    noise = torch.randn_like(wav)
    sig_power = wav.pow(2).mean().clamp_min(1e-8)
    noise_power = noise.pow(2).mean().clamp_min(1e-8)
    k = torch.sqrt(sig_power / (noise_power * (10 ** (snr_db / 10.0))))
    return (wav + k * noise).clamp(-1.0, 1.0)

def _convolve_rir(wav: torch.Tensor, rir_path: str) -> torch.Tensor:
    rir, _ = torchaudio.load(rir_path)
    rir = rir.mean(dim=0, keepdim=True)
    rir = rir / (rir.abs().max().clamp_min(1e-8))
    out = torchaudio.functional.fftconvolve(wav, rir)
    out = out[..., : wav.shape[-1]]
    return out.clamp(-1.0, 1.0)

def _apply_codec_ffmpeg(wav_np: np.ndarray, sr: int, codec: str, bitrate: int) -> np.ndarray:
    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        in_wav = os.path.join(td, "in.wav")
        mid = os.path.join(td, "mid")
        out_wav = os.path.join(td, "out.wav")
        sf.write(in_wav, wav_np.astype(np.float32), sr)

        if codec == "opus":
            mid_file = mid + ".opus"
            cmd1 = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", in_wav, "-c:a", "libopus", "-b:a", str(bitrate), mid_file]
            cmd2 = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", mid_file, out_wav]
        elif codec == "mp3":
            mid_file = mid + ".mp3"
            cmd1 = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", in_wav, "-c:a", "libmp3lame", "-b:a", str(bitrate), mid_file]
            cmd2 = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", mid_file, out_wav]
        else:
            raise ValueError(f"Unknown codec: {codec}")

        subprocess.check_call(cmd1)
        subprocess.check_call(cmd2)
        y, _ = sf.read(out_wav, always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        return y.astype(np.float32)

class AudioCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        sr: int,
        clip_seconds: float,
        vad: bool = False,
        augment: Optional[AugmentConfig] = None,
        features: bool = True,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        num_classes: int = 2,
    ):
        self.df = pd.read_csv(csv_path)
        self.sr = sr
        self.clip_seconds = clip_seconds
        self.vad = vad
        self.augment = augment or AugmentConfig(enable=False)
        self.features = features
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_classes = num_classes
        if "class_id" not in self.df.columns:
            self.df["class_id"] = self.df["label"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row["path"])
        y = int(row["class_id"])

        clip = load_wav(path, target_sr=self.sr, mono=True)
        wav = clip.wav
        if self.vad:
            wav = simple_vad_trim(wav, self.sr)
        wav = pad_or_crop(wav, self.sr, self.clip_seconds, random_crop=True)

        if self.augment.enable:
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            if random.random() < self.augment.noise_prob:
                snr = random.uniform(self.augment.noise_snr_db_min, self.augment.noise_snr_db_max)
                wav_t = _add_noise(wav_t, snr)
            if self.augment.rir_dir and random.random() < self.augment.reverb_prob:
                rirs = [os.path.join(self.augment.rir_dir, p) for p in os.listdir(self.augment.rir_dir) if p.lower().endswith(".wav")]
                if rirs:
                    wav_t = _convolve_rir(wav_t, random.choice(rirs))
            wav = wav_t.squeeze(0).cpu().numpy().astype(np.float32)

            if random.random() < self.augment.codec_prob:
                try:
                    wav = _apply_codec_ffmpeg(wav, self.sr, self.augment.codec, self.augment.codec_bitrate)
                    wav = pad_or_crop(wav, self.sr, self.clip_seconds, random_crop=False)
                except Exception:
                    pass

        if self.features:
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            feat = mel_spectrogram_db(wav_t, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
            feat = feat.unsqueeze(1)  # (1,1,n_mels,T)
            return feat.squeeze(0), torch.tensor(y, dtype=torch.long)

        return torch.from_numpy(wav).float(), torch.tensor(y, dtype=torch.long)
