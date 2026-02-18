from __future__ import annotations

import os
import json
import glob
import time
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import yaml

from scipy.signal import butter, lfilter, resample_poly, fftconvolve

from v2v_safety.audio import load_wav
from v2v_safety.watermark import (
    WatermarkConfig,
    embed_zero_bit_spread_spectrum,
    detect_zero_bit_spread_spectrum,
)


# ---------------- utils ----------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return d or {}

def ensure_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x

def rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(x * x) + eps))

def set_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    rc = rms(clean)
    rn = rms(noise)
    if rn < 1e-12:
        return clean
    target_rn = rc / (10 ** (snr_db / 20.0))
    return clean + noise * (target_rn / rn)

def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None

def ffmpeg_codec_roundtrip(x: np.ndarray, sr: int, codec: str, bitrate: str) -> np.ndarray:
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not found in PATH (needed for codec attacks).")

    x = ensure_mono(x)

    with tempfile.TemporaryDirectory() as td:
        in_wav = os.path.join(td, "in.wav")
        out_enc = os.path.join(td, f"out.{codec}")
        out_wav = os.path.join(td, "out.wav")

        sf.write(in_wav, x, sr, subtype="PCM_16")

        if codec == "mp3":
            cmd1 = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", in_wav, "-b:a", bitrate, out_enc]
        elif codec == "aac":
            cmd1 = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", in_wav, "-c:a", "aac", "-b:a", bitrate, out_enc]
        else:
            raise ValueError(codec)

        subprocess.check_call(cmd1)

        cmd2 = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", out_enc, "-ar", str(sr), "-ac", "1", out_wav]
        subprocess.check_call(cmd2)

        y, sr2 = sf.read(out_wav, dtype="float32", always_2d=False)
        y = ensure_mono(y)
        if sr2 != sr:
            y = resample_poly(y, sr, sr2).astype(np.float32)
        return y.astype(np.float32)


# ---------------- attacks as specs (picklable) ----------------

@dataclass(frozen=True)
class AttackSpec:
    name: str
    p: Dict[str, Any]

def build_attacks(seed: int, include_codecs: bool) -> List[AttackSpec]:
    attacks: List[AttackSpec] = [
        AttackSpec("none", {}),
        AttackSpec("gain", {"db": +6.0}),
        AttackSpec("gain", {"db": -6.0}),
        AttackSpec("clip", {"level": 0.8}),
        AttackSpec("clip", {"level": 0.6}),
        AttackSpec("resample_roundtrip", {"target_sr": 16000}),
        AttackSpec("resample_roundtrip", {"target_sr": 8000}),
        AttackSpec("lowpass", {"cutoff_hz": 7000.0}),
        AttackSpec("lowpass", {"cutoff_hz": 3400.0}),
        AttackSpec("white_noise", {"snr_db": 30.0, "seed": seed}),
        AttackSpec("white_noise", {"snr_db": 20.0, "seed": seed}),
        AttackSpec("white_noise", {"snr_db": 10.0, "seed": seed}),
        AttackSpec("white_noise", {"snr_db": 5.0, "seed": seed}),
        AttackSpec("reverb_synth", {"rt60": 0.2, "seed": seed}),
        AttackSpec("reverb_synth", {"rt60": 0.4, "seed": seed}),
    ]

    if include_codecs and ffmpeg_exists():
        attacks += [
            AttackSpec("mp3_roundtrip", {"bitrate": "320k"}),
            AttackSpec("mp3_roundtrip", {"bitrate": "128k"}),
            AttackSpec("mp3_roundtrip", {"bitrate": "64k"}),
            AttackSpec("mp3_roundtrip", {"bitrate": "32k"}),
            AttackSpec("aac_roundtrip", {"bitrate": "128k"}),
            AttackSpec("aac_roundtrip", {"bitrate": "64k"}),
        ]
    return attacks


# ---------------- worker globals ----------------

_G_WCFG: Optional[WatermarkConfig] = None
_G_ATTACKS: Optional[List[AttackSpec]] = None

def _init_worker(wcfg_dict: Dict[str, Any], attacks: List[AttackSpec]):
    # важно: чтобы не было оверсабскрайба (много процессов * много потоков BLAS)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    global _G_WCFG, _G_ATTACKS
    _G_WCFG = WatermarkConfig(**wcfg_dict)
    _G_ATTACKS = attacks

def _butter_lowpass_coeff(sr: int, cutoff_hz: float, order: int = 6):
    nyq = 0.5 * sr
    wn = cutoff_hz / nyq
    return butter(order, wn, btype="lowpass")

def _apply_attack(x: np.ndarray, sr: int, spec: AttackSpec,
                  cache: Dict[Tuple, Any]) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    name = spec.name
    p = spec.p

    x = ensure_mono(x)

    if name == "none":
        return x, sr, {"attack": "none"}

    if name == "gain":
        db = float(p["db"])
        g = 10 ** (db / 20.0)
        return (x * g).astype(np.float32), sr, {"attack": "gain", "db": db}

    if name == "clip":
        level = float(p["level"])
        return np.clip(x, -level, level).astype(np.float32), sr, {"attack": "clip", "level": level}

    if name == "resample_roundtrip":
        target_sr = int(p["target_sr"])
        y_ds = resample_poly(x, target_sr, sr).astype(np.float32)
        y_us = resample_poly(y_ds, sr, target_sr).astype(np.float32)
        return y_us, sr, {"attack": "resample_roundtrip", "target_sr": target_sr}

    if name == "lowpass":
        cutoff_hz = float(p["cutoff_hz"])
        key = ("lowpass", sr, cutoff_hz)
        if key not in cache:
            cache[key] = _butter_lowpass_coeff(sr, cutoff_hz, order=6)
        b, a = cache[key]
        y = lfilter(b, a, x).astype(np.float32)
        return y, sr, {"attack": "lowpass", "cutoff_hz": cutoff_hz}

    if name == "white_noise":
        snr_db = float(p["snr_db"])
        seed = int(p.get("seed", 0))
        rng = np.random.default_rng(seed)
        n = rng.standard_normal(len(x)).astype(np.float32)
        y = set_snr(x, n, snr_db).astype(np.float32)
        return y, sr, {"attack": "white_noise", "snr_db": snr_db}

    if name == "reverb_synth":
        rt60 = float(p["rt60"])
        seed = int(p.get("seed", 0))
        key = ("ir", sr, rt60, seed)
        if key not in cache:
            rng = np.random.default_rng(seed)
            ir_len = int(sr * min(1.2, max(0.3, rt60 * 3.0)))
            t = np.arange(ir_len) / sr
            decay = np.exp(-t / max(1e-3, rt60)).astype(np.float32)
            ir = decay.copy()
            for _ in range(6):
                d = rng.integers(low=int(0.005 * sr), high=int(0.08 * sr))
                if d < ir_len:
                    ir[d] += float(rng.uniform(0.1, 0.6))
            ir = (ir / (np.sum(np.abs(ir)) + 1e-12)).astype(np.float32)
            cache[key] = ir
        ir = cache[key]
        out = fftconvolve(x, ir, mode="full")[: len(x)].astype(np.float32)
        return out, sr, {"attack": "reverb_synth", "rt60": rt60}

    if name == "mp3_roundtrip":
        bitrate = str(p["bitrate"])
        y = ffmpeg_codec_roundtrip(x, sr, "mp3", bitrate)
        return y, sr, {"attack": "mp3_roundtrip", "bitrate": bitrate}

    if name == "aac_roundtrip":
        bitrate = str(p["bitrate"])
        y = ffmpeg_codec_roundtrip(x, sr, "aac", bitrate)
        return y, sr, {"attack": "aac_roundtrip", "bitrate": bitrate}

    raise ValueError(f"Unknown attack: {name}")


def _process_one(args: Tuple[int, str]) -> List[Dict[str, Any]]:
    idx, path = args
    assert _G_WCFG is not None and _G_ATTACKS is not None
    wcfg = _G_WCFG
    attacks = _G_ATTACKS

    cache: Dict[Tuple, Any] = {}

    clip = load_wav(path, target_sr=int(wcfg.sr))
    x = ensure_mono(clip.wav)

    x_wm = embed_zero_bit_spread_spectrum(x, wcfg)

    rows: List[Dict[str, Any]] = []
    for label, sig in [("clean", x), ("wm", x_wm)]:
        for spec in attacks:
            y, ysr, meta = _apply_attack(sig, int(wcfg.sr), spec, cache)

            if ysr != int(wcfg.sr):
                y = resample_poly(y, int(wcfg.sr), ysr).astype(np.float32)
                ysr = int(wcfg.sr)

            present, score = detect_zero_bit_spread_spectrum(y, wcfg)

            rows.append({
                "file": os.path.basename(path),
                "idx": idx,
                "label": label,
                "is_watermarked": int(label == "wm"),
                "det_present": int(bool(present)),
                "score": float(score),
                "threshold": float(wcfg.threshold),
                "sr": int(ysr),
                "dur_s": float(len(y) / ysr),
                **meta,
            })
    return rows


# ---------------- benchmark ----------------

def run_benchmark(
    in_dir: str,
    watermark_cfg_path: str = "configs/watermark.yaml",
    out_csv: str = "results.csv",
    out_json: str = "summary.json",
    limit: int = 0,
    seed: int = 0,
    workers: int = 0,
    include_codecs: bool = True,
):
    wcfg_dict = load_yaml(watermark_cfg_path)

    wav_paths = sorted(glob.glob(os.path.join(in_dir, "*.wav")))
    if limit and limit > 0:
        wav_paths = wav_paths[:limit]
    if not wav_paths:
        raise RuntimeError(f"No .wav found in {in_dir}")

    attacks = build_attacks(seed=seed, include_codecs=include_codecs)

    if workers is None or workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)

    t0 = time.time()
    rows: List[Dict[str, Any]] = []

    # ВАЖНО: ffmpeg атаки могут забить диск/CPU если workers слишком много
    if include_codecs and ffmpeg_exists() and workers > 8:
        print("WARN: include_codecs=True и workers>8 на Windows может замедлить из-за множества ffmpeg процессов.")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(wcfg_dict, attacks),
    ) as ex:
        futures = [ex.submit(_process_one, (i, p)) for i, p in enumerate(wav_paths)]
        done = 0
        for fut in as_completed(futures):
            rows.extend(fut.result())
            done += 1
            if done % 10 == 0:
                print(f"Processed {done}/{len(wav_paths)}")

    import csv
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    # summary
    wcfg = WatermarkConfig(**wcfg_dict)
    thr = float(wcfg.threshold)
    summary = {"threshold": thr, "n_files": len(wav_paths), "elapsed_s": time.time() - t0, "by_attack": {}}

    by: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], List[Dict[str, Any]]] = {}
    keep_params = ("bitrate", "target_sr", "cutoff_hz", "snr_db", "rt60", "db", "level")
    for r in rows:
        params = tuple(sorted((k, r[k]) for k in keep_params if k in r))
        k = (r["attack"], params)
        by.setdefault(k, []).append(r)

    for (atk_name, atk_params), rs in by.items():
        wm = [x for x in rs if x["is_watermarked"] == 1]
        cl = [x for x in rs if x["is_watermarked"] == 0]
        tpr = float(np.mean([1.0 if x["score"] >= thr else 0.0 for x in wm])) if wm else None
        fpr = float(np.mean([1.0 if x["score"] >= thr else 0.0 for x in cl])) if cl else None

        summary["by_attack"][f"{atk_name} {dict(atk_params)}"] = {
            "n": len(rs),
            "tpr": tpr,
            "fpr": fpr,
            "mean_score_wm": float(np.mean([x["score"] for x in wm])) if wm else None,
            "mean_score_clean": float(np.mean([x["score"] for x in cl])) if cl else None,
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("wm robustness benchmark (parallel)")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--watermark-cfg", default="configs/watermark.yaml")
    ap.add_argument("--out-csv", default="results.csv")
    ap.add_argument("--out-json", default="summary.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="0 -> cpu_count-1")
    ap.add_argument("--no-codecs", action="store_true", help="skip mp3/aac attacks (ffmpeg)")
    args = ap.parse_args()

    run_benchmark(
        in_dir=args.in_dir,
        watermark_cfg_path=args.watermark_cfg,
        out_csv=args.out_csv,
        out_json=args.out_json,
        limit=args.limit,
        seed=args.seed,
        workers=args.workers,
        include_codecs=not args.no_codecs,
    )
