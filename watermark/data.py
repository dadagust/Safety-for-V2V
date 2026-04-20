from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset


AUDIO_EXTS_DEFAULT = ("wav", "flac", "mp3", "ogg", "m4a", "aac")


def list_audio_files(audio_dir: str | Path, exts: Iterable[str] = AUDIO_EXTS_DEFAULT) -> List[str]:
    audio_dir = Path(audio_dir)
    exts_l = {e.lower().lstrip(".") for e in exts}
    paths: List[str] = []
    for p in audio_dir.rglob("*"):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts_l:
            paths.append(str(p))
    paths.sort()
    return paths


def write_manifest(paths: List[str], out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path"])
        for p in paths:
            w.writerow([p])


def read_manifest(manifest_csv: str | Path) -> List[str]:
    manifest_csv = Path(manifest_csv)
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "path" not in r.fieldnames:
            raise ValueError(f"Manifest must have a 'path' column. Got: {r.fieldnames}")
        return [row["path"] for row in r]


def _to_mono(x: np.ndarray) -> np.ndarray:
    # x: [T] or [T, C]
    if x.ndim == 1:
        return x.astype(np.float32)
    if x.ndim == 2:
        return x.mean(axis=1).astype(np.float32)
    raise ValueError(f"Unexpected audio shape: {x.shape}")


def _resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x.astype(np.float32)
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Bad sample rates: orig_sr={orig_sr}, target_sr={target_sr}")
    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    y = resample_poly(x, up, down)
    return y.astype(np.float32)


def load_audio(path: str | Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Returns mono float32 waveform in [-1, 1] (best-effort) and sample_rate.

    The manifests in this project were originally created on Windows, so some
    paths contain backslashes. For portability we normalize separators when the
    raw path does not exist on the current platform.
    """
    path = str(path)
    if not Path(path).exists() and "\\" in path:
        alt = path.replace("\\", "/")
        if Path(alt).exists():
            path = alt
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    x = _to_mono(np.asarray(x))
    # Some formats may produce >1.0 peaks; clip gently.
    x = np.clip(x, -1.0, 1.0)
    if sr != target_sr:
        x = _resample(x, sr, target_sr)
        sr = target_sr
    return x, sr


def pad_or_trim(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] == target_len:
        return x
    if x.shape[0] < target_len:
        pad = target_len - x.shape[0]
        return np.pad(x, (0, pad), mode="constant")
    return x[:target_len]


class AudioManifestDataset(Dataset):
    """
    Loads audio paths from a CSV manifest and returns fixed-length segments.
    """
    def __init__(
        self,
        manifest_csv: str | Path,
        sample_rate: int = 16000,
        segment_seconds: float = 4.0,
        random_crop: bool = True,
    ) -> None:
        self.paths = read_manifest(manifest_csv)
        if len(self.paths) == 0:
            raise ValueError(f"No paths found in manifest: {manifest_csv}")
        self.sample_rate = int(sample_rate)
        self.segment_len = int(round(segment_seconds * self.sample_rate))
        self.random_crop = bool(random_crop)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        x, _sr = load_audio(path, self.sample_rate)

        if x.shape[0] < self.segment_len:
            x = pad_or_trim(x, self.segment_len)
        elif self.random_crop and x.shape[0] > self.segment_len:
            max_start = x.shape[0] - self.segment_len
            start = np.random.randint(0, max_start + 1)
            x = x[start : start + self.segment_len]
        else:
            x = x[: self.segment_len]

        # torch tensor [1, T]
        xt = torch.from_numpy(x).unsqueeze(0)
        return {
            "audio": xt,
            "path": path,
        }
