from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
import yaml

from watermark.data import load_audio, read_manifest
from watermark.runtime import build_model_from_cfg, embed_batch


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(cfg: Dict[str, Any], checkpoint_path: str, device: torch.device):
    model = build_model_from_cfg(cfg, device=device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def resolve_input_path(input_path: str, manifest: str, index: int) -> str:
    if input_path:
        return input_path
    if manifest:
        paths = read_manifest(manifest)
        if len(paths) == 0:
            raise ValueError(f"Empty manifest: {manifest}")
        index = max(0, min(index, len(paths) - 1))
        return paths[index]
    raise ValueError("Provide either --input or --manifest")


def parse_bits(bits_arg: str, nbits: int, device: torch.device) -> torch.Tensor:
    if bits_arg.lower() == "random":
        return torch.randint(0, 2, (1, nbits), device=device, dtype=torch.float32)
    clean = bits_arg.strip().replace(" ", "")
    if len(clean) != nbits or any(c not in "01" for c in clean):
        raise ValueError(f"bits must be a binary string of length {nbits} or 'random'")
    arr = torch.tensor([[float(c) for c in clean]], device=device, dtype=torch.float32)
    return arr


def normalize_for_listening(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x)))
    if m < 1e-8:
        return x.astype(np.float32)
    return (x * (peak / m)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--manifest", type=str, default="")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--embed_mode", type=str, default="auto", choices=["auto", "full", "window", "dropout"])
    ap.add_argument("--bits", type=str, default="random", help="Binary string or 'random'")
    ap.add_argument("--delta_gain", type=float, default=20.0, help="Gain for audible residual example")
    ap.add_argument("--segment_seconds_override", type=float, default=0.0, help="If >0, crop/pad to this duration instead of config segment_seconds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    model = load_model(cfg, args.checkpoint, device)
    sr = int(cfg["audio"]["sample_rate"])
    seg_seconds = float(args.segment_seconds_override) if float(args.segment_seconds_override) > 0 else float(cfg["audio"]["segment_seconds"])
    seg_len = int(round(seg_seconds * sr))
    wm_len = int(round(float(cfg["audio"]["watermark_seconds"]) * sr))
    nbits = int(cfg["model"]["nbits"])

    input_path = resolve_input_path(args.input, args.manifest, args.index)
    x_np, _ = load_audio(input_path, target_sr=sr)
    if x_np.shape[0] < seg_len:
        x_np = np.pad(x_np, (0, seg_len - x_np.shape[0]), mode="constant")
    else:
        x_np = x_np[:seg_len]

    x = torch.from_numpy(x_np).to(device).view(1, 1, -1)
    bits = parse_bits(args.bits, nbits, device)

    embed_mode = args.embed_mode
    if embed_mode == "auto":
        mode = str((cfg.get("embed", {}) or {}).get("mode", "window")).lower()
        embed_mode = "dropout" if mode in ("dropout_holes", "holes") else mode
        if embed_mode not in ("full", "window", "dropout"):
            embed_mode = "window"

    with torch.no_grad():
        y, mask = embed_batch(
            x=x,
            bits=bits,
            model=model,
            cfg=cfg,
            sample_rate=sr,
            watermark_len=wm_len,
            p_embed=1.0,
            embed_mode_override=embed_mode,
        )

    x_out = x.squeeze().detach().cpu().numpy().astype(np.float32)
    y_out = y.squeeze().detach().cpu().numpy().astype(np.float32)
    delta = (y_out - x_out).astype(np.float32)
    delta_amp = np.clip(delta * float(args.delta_gain), -1.0, 1.0).astype(np.float32)
    delta_norm = normalize_for_listening(delta)
    mask_np = mask.squeeze().detach().cpu().numpy().astype(np.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "original.wav", x_out, sr)
    sf.write(out_dir / "watermarked.wav", y_out, sr)
    sf.write(out_dir / "delta.wav", delta, sr)
    sf.write(out_dir / "delta_normalized.wav", delta_norm, sr)
    sf.write(out_dir / "delta_xgain.wav", delta_amp, sr)

    meta = {
        "input_path": input_path,
        "sample_rate": sr,
        "segment_seconds": seg_seconds,
        "embed_mode": embed_mode,
        "bits": "".join(str(int(v)) for v in bits[0].detach().cpu().tolist()),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "watermark_fraction": float(mask_np.mean()),
        "delta_gain_for_preview": float(args.delta_gain),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote listening examples to: {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
