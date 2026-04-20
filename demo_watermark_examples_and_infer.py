from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
import yaml

from watermark.data import load_audio, read_manifest, pad_or_trim
from watermark.runtime import (
    build_model_from_cfg,
    detector_forward_with_predicted_mask,
    embed_batch,
    present_from_presence_logits,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(cfg: Dict[str, Any], checkpoint_path: str, device: torch.device):
    model = build_model_from_cfg(cfg, device=device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def sanitize_name(name: str) -> str:
    name = Path(name).stem
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:80] if len(name) > 80 else name


def normalize_for_listening(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x)))
    if m < 1e-8:
        return x.astype(np.float32)
    return (x * (peak / m)).astype(np.float32)


def parse_bits_arg(bits_arg: str, nbits: int, device: torch.device) -> torch.Tensor:
    if bits_arg.lower() == "random":
        return torch.randint(0, 2, (1, nbits), device=device, dtype=torch.float32)
    clean = bits_arg.strip().replace(" ", "")
    if len(clean) != nbits or any(c not in "01" for c in clean):
        raise ValueError(f"bits must be a binary string of length {nbits} or 'random'; got {bits_arg!r}")
    return torch.tensor([[float(c) for c in clean]], device=device, dtype=torch.float32)


def resolve_embed_mode(cfg: Dict[str, Any], arg_mode: str) -> str:
    if arg_mode != "auto":
        return arg_mode
    mode = str((cfg.get("embed", {}) or {}).get("mode", "window")).lower()
    if mode in ("full", "window", "dropout", "dropout_holes", "holes"):
        return "dropout" if mode in ("dropout_holes", "holes") else mode
    return "window"


@torch.no_grad()
def infer_one(
    model,
    audio_t: torch.Tensor,
    threshold: float,
    min_positive_fraction: float,
    use_hard_mask: bool,
) -> Dict[str, Any]:
    presence_logits, bit_logits, weights = detector_forward_with_predicted_mask(
        model.detector,
        audio_t,
        threshold=threshold,
        use_soft_mask=not use_hard_mask,
    )
    pred_present = bool(
        present_from_presence_logits(
            presence_logits,
            threshold=threshold,
            min_positive_fraction=min_positive_fraction,
        )[0].item()
    )
    presence_prob = torch.sigmoid(presence_logits)
    pred_bits = (torch.sigmoid(bit_logits) >= 0.5).float()
    return {
        "pred_present": pred_present,
        "pred_bits": "".join(str(int(v)) for v in pred_bits[0].cpu().tolist()),
        "mean_presence_prob": float(presence_prob.mean().item()),
        "max_presence_prob": float(presence_prob.max().item()),
        "positive_fraction": float((presence_prob >= threshold).float().mean().item()),
        "mask_weight_mean": float(weights.mean().item()),
        "mask_weight_max": float(weights.max().item()),
    }


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x.astype(np.float32), sr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample a few audio files, render original/watermarked examples, and run inference on both.",
    )
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_examples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--embed_mode", type=str, default="auto", choices=["auto", "full", "window", "dropout"])
    ap.add_argument("--bits", type=str, default="random", help="Binary string or 'random'.")
    ap.add_argument("--delta_gain", type=float, default=20.0, help="Gain for audible residual preview.")
    ap.add_argument("--use_hard_mask", action="store_true", help="Use thresholded predicted mask for bit decoding.")
    ap.add_argument("--segment_seconds_override", type=float, default=0.0, help="If >0, crop/pad to this duration instead of config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)

    sr = int(cfg["audio"]["sample_rate"])
    seg_seconds = float(args.segment_seconds_override) if float(args.segment_seconds_override) > 0 else float(cfg["audio"]["segment_seconds"])
    seg_len = int(round(seg_seconds * sr))
    wm_len = int(round(float(cfg["audio"]["watermark_seconds"]) * sr))
    nbits = int(cfg["model"]["nbits"])
    threshold = float(cfg.get("detection", {}).get("presence_threshold", 0.5))
    min_pos_frac = float(cfg.get("detection", {}).get("min_positive_fraction", 0.05))

    model = load_model(cfg, args.checkpoint, device)
    embed_mode = resolve_embed_mode(cfg, args.embed_mode)

    paths = read_manifest(args.manifest)
    if len(paths) == 0:
        raise ValueError(f"Manifest is empty: {args.manifest}")

    rng = random.Random(args.seed)
    sample_n = min(int(args.num_examples), len(paths))
    chosen = rng.sample(paths, k=sample_n)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for i, path in enumerate(chosen, start=1):
        x_np, _ = load_audio(path, target_sr=sr)
        x_np = pad_or_trim(x_np, seg_len).astype(np.float32)
        x = torch.from_numpy(x_np).to(device).view(1, 1, -1)
        bits = parse_bits_arg(args.bits, nbits, device)

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

        x_out = x.squeeze().cpu().numpy().astype(np.float32)
        y_out = y.squeeze().cpu().numpy().astype(np.float32)
        delta = (y_out - x_out).astype(np.float32)
        delta_norm = normalize_for_listening(delta)
        delta_xgain = np.clip(delta * float(args.delta_gain), -1.0, 1.0).astype(np.float32)
        mask_np = mask.squeeze().cpu().numpy().astype(np.float32)
        gt_bits = "".join(str(int(v)) for v in bits[0].cpu().tolist())

        pred_orig = infer_one(model, x, threshold, min_pos_frac, args.use_hard_mask)
        pred_wm = infer_one(model, y, threshold, min_pos_frac, args.use_hard_mask)

        wm_pred_bits_arr = np.array([int(c) for c in pred_wm["pred_bits"]], dtype=np.int64)
        gt_bits_arr = np.array([int(c) for c in gt_bits], dtype=np.int64)
        bit_acc = float((wm_pred_bits_arr == gt_bits_arr).mean())
        msg_correct = bool(np.all(wm_pred_bits_arr == gt_bits_arr))

        sample_dir = out_dir / f"sample_{i:02d}_{sanitize_name(path)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_wav(sample_dir / "original.wav", x_out, sr)
        write_wav(sample_dir / "watermarked.wav", y_out, sr)
        write_wav(sample_dir / "delta.wav", delta, sr)
        write_wav(sample_dir / "delta_normalized.wav", delta_norm, sr)
        write_wav(sample_dir / "delta_xgain.wav", delta_xgain, sr)

        meta = {
            "source_path": path,
            "sample_rate": sr,
            "segment_seconds": seg_seconds,
            "embed_mode": embed_mode,
            "gt_bits": gt_bits,
            "watermark_fraction": float(mask_np.mean()),
            "max_abs_delta": float(np.max(np.abs(delta))),
            "mean_abs_delta": float(np.mean(np.abs(delta))),
            "delta_gain_for_preview": float(args.delta_gain),
            "inference_original": pred_orig,
            "inference_watermarked": pred_wm,
            "watermarked_bit_acc": bit_acc,
            "watermarked_msg_correct": msg_correct,
        }
        (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        rows.append({
            "sample_id": i,
            "source_path": path,
            "embed_mode": embed_mode,
            "gt_bits": gt_bits,
            "wm_fraction": float(mask_np.mean()),
            "orig_pred_present": pred_orig["pred_present"],
            "orig_pred_bits": pred_orig["pred_bits"],
            "orig_mean_presence_prob": pred_orig["mean_presence_prob"],
            "orig_positive_fraction": pred_orig["positive_fraction"],
            "wm_pred_present": pred_wm["pred_present"],
            "wm_pred_bits": pred_wm["pred_bits"],
            "wm_mean_presence_prob": pred_wm["mean_presence_prob"],
            "wm_positive_fraction": pred_wm["positive_fraction"],
            "watermarked_bit_acc": bit_acc,
            "watermarked_msg_correct": msg_correct,
            "max_abs_delta": float(np.max(np.abs(delta))),
            "mean_abs_delta": float(np.mean(np.abs(delta))),
            "sample_dir": str(sample_dir),
        })

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(rows)} examples to: {out_dir}")
    print(f"Summary CSV: {out_dir / 'summary.csv'}")
    print(f"Summary JSON: {out_dir / 'summary.json'}")
    for row in rows:
        print(
            f"[{row['sample_id']}] orig_present={row['orig_pred_present']} wm_present={row['wm_pred_present']} "
            f"wm_bit_acc={row['watermarked_bit_acc']:.3f} msg_ok={row['watermarked_msg_correct']} "
            f"path={row['source_path']}"
        )


if __name__ == "__main__":
    main()
