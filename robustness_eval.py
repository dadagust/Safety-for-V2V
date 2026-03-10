from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
import yaml

from watermark.attacks import (
    aac_roundtrip_np,
    add_noise_snr_np,
    bitdepth_reduce_np,
    crop_pad_np,
    lowpass_np,
    mp3_roundtrip_np,
    resample_speed_np,
)
from watermark.data import load_audio
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


def read_manifest_paths(manifest_csv: str) -> List[str]:
    with open(manifest_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row["path"] for row in r]


def resolve_embed_mode(cfg: Dict[str, Any], arg_mode: str) -> str:
    if arg_mode != "auto":
        return arg_mode
    mode = str((cfg.get("embed", {}) or {}).get("mode", "window")).lower()
    if mode in ("full", "window", "dropout", "dropout_holes", "holes"):
        return "dropout" if mode in ("dropout_holes", "holes") else mode
    return "window"


def detect_and_decode(
    detector,
    audio_t: torch.Tensor,
    presence_threshold: float,
    min_positive_fraction: float,
    use_soft_mask: bool,
) -> Tuple[bool, torch.Tensor, float]:
    with torch.no_grad():
        presence_logits, bit_logits, _weights = detector_forward_with_predicted_mask(
            detector,
            audio_t,
            threshold=presence_threshold,
            use_soft_mask=use_soft_mask,
        )
        prob = torch.sigmoid(presence_logits)[0]
        present = bool(present_from_presence_logits(presence_logits, presence_threshold, min_positive_fraction)[0].item())
        presence_score = float(prob.max().item())
        return present, bit_logits[0], presence_score


def build_attack_cases(cfg: Dict[str, Any], sr: int) -> List[Tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    cases: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
    cases.append(("clean", lambda x: x))

    for snr in cfg["attacks_eval"]["noise_snr_db_list"]:
        cases.append((f"noise_snr{snr}", lambda x, snr=snr: add_noise_snr_np(x, float(snr))))

    for f in cfg["attacks_eval"]["resample_factor_list"]:
        cases.append((f"speed_{f}", lambda x, f=f: resample_speed_np(x, float(f), sr)))

    for c in cfg["attacks_eval"]["lowpass_hz_list"]:
        cases.append((f"lowpass_{c}", lambda x, c=c: lowpass_np(x, sr, float(c))))

    for cs in cfg["attacks_eval"]["crop_seconds_list"]:
        cases.append((f"crop_{cs}s", lambda x, cs=cs: crop_pad_np(x, float(cs), sr)))

    for b in cfg["attacks_eval"]["bitdepth_list"]:
        cases.append((f"bitdepth_{b}", lambda x, b=b: bitdepth_reduce_np(x, int(b))))

    for br in cfg["attacks_eval"]["mp3_bitrates_k"]:
        cases.append((f"mp3_{br}k", lambda x, br=br: mp3_roundtrip_np(x, sr, int(br))))
    for br in cfg["attacks_eval"]["aac_bitrates_k"]:
        cases.append((f"aac_{br}k", lambda x, br=br: aac_roundtrip_np(x, sr, int(br))))

    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max_items", type=int, default=200, help="Limit items for speed. 0 = all.")
    ap.add_argument("--latency", action="store_true", help="Also compute detection/msg accuracy vs chunk duration.")
    ap.add_argument("--embed_mode", type=str, default="auto", choices=["auto", "full", "window", "dropout"])
    ap.add_argument("--use_hard_mask", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    sr = int(cfg["audio"]["sample_rate"])
    seg_len = int(round(cfg["audio"]["segment_seconds"] * sr))
    wm_len = int(round(float(cfg["audio"]["watermark_seconds"]) * sr))

    model = load_model(cfg, args.checkpoint, device)
    presence_thr = float(cfg["detection"]["presence_threshold"])
    min_frac = float(cfg["detection"]["min_positive_fraction"])
    use_soft_mask = not args.use_hard_mask
    embed_mode = resolve_embed_mode(cfg, args.embed_mode)

    paths = read_manifest_paths(args.manifest)
    if args.max_items and args.max_items > 0:
        paths = paths[: args.max_items]

    attack_cases = build_attack_cases(cfg, sr)
    chunk_secs = list(cfg["latency_probe"]["chunk_seconds"])

    stats = {}
    for name, _fn in attack_cases:
        stats[name] = {
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "bit_correct": 0, "bit_total": 0,
            "msg_correct": 0, "msg_total": 0,
        }
        if args.latency:
            for cs in chunk_secs:
                stats[name][f"tp@{cs}s"] = 0
                stats[name][f"msg_correct@{cs}s"] = 0
                stats[name][f"msg_total@{cs}s"] = 0

    nbits = int(cfg["model"]["nbits"])

    for path in tqdm(paths, desc="robustness"):
        x_np, _ = load_audio(path, target_sr=sr)
        if x_np.shape[0] < seg_len:
            x_np = np.pad(x_np, (0, seg_len - x_np.shape[0]), mode="constant")
        else:
            x_np = x_np[:seg_len]

        x = torch.from_numpy(x_np).to(device).view(1, 1, -1)
        bits = torch.randint(0, 2, (1, nbits), device=device, dtype=torch.float32)

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
        y_np = y.detach().cpu().numpy().reshape(-1).astype(np.float32)
        x0_np = x_np.astype(np.float32)

        for name, fn in attack_cases:
            try:
                y_att_pos = fn(y_np)
                y_att_neg = fn(x0_np)
            except subprocess.CalledProcessError:
                continue

            y_pos_t = torch.from_numpy(y_att_pos).to(device).view(1, 1, -1)
            y_neg_t = torch.from_numpy(y_att_neg).to(device).view(1, 1, -1)

            pos_present, pos_bits_logits, _ = detect_and_decode(model.detector, y_pos_t, presence_thr, min_frac, use_soft_mask)
            neg_present, _neg_bits_logits, _ = detect_and_decode(model.detector, y_neg_t, presence_thr, min_frac, use_soft_mask)

            if pos_present:
                stats[name]["tp"] += 1
            else:
                stats[name]["fn"] += 1

            if neg_present:
                stats[name]["fp"] += 1
            else:
                stats[name]["tn"] += 1

            if pos_present:
                pred_bits = (torch.sigmoid(pos_bits_logits) >= 0.5).float()
                stats[name]["bit_correct"] += int((pred_bits == bits[0]).sum().item())
                stats[name]["bit_total"] += nbits
                stats[name]["msg_correct"] += int(bool(torch.all(pred_bits == bits[0]).item()))
                stats[name]["msg_total"] += 1

            if args.latency:
                for cs in chunk_secs:
                    Tchunk = max(64, int(round(float(cs) * sr)))
                    y_chunk = y_att_pos[:Tchunk]
                    y_chunk_t = torch.from_numpy(y_chunk).to(device).view(1, 1, -1)
                    present_c, bits_logits_c, _ = detect_and_decode(model.detector, y_chunk_t, presence_thr, min_frac, use_soft_mask)
                    if present_c:
                        stats[name][f"tp@{cs}s"] += 1
                        pred_bits_c = (torch.sigmoid(bits_logits_c) >= 0.5).float()
                        stats[name][f"msg_correct@{cs}s"] += int(bool(torch.all(pred_bits_c == bits[0]).item()))
                    stats[name][f"msg_total@{cs}s"] += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, s in stats.items():
        tp, fp, tn, fn = s["tp"], s["fp"], s["tn"], s["fn"]
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        bit_acc = s["bit_correct"] / max(s["bit_total"], 1)
        msg_acc = s["msg_correct"] / max(s["msg_total"], 1)

        row = {
            "attack": name,
            "TPR": tpr,
            "FPR": fpr,
            "bit_acc": bit_acc,
            "msg_acc": msg_acc,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        }
        if args.latency:
            for cs in chunk_secs:
                row[f"TPR@{cs}s"] = s[f"tp@{cs}s"] / max(s["tp"] + s["fn"], 1)
                row[f"msg_acc@{cs}s"] = s[f"msg_correct@{cs}s"] / max(s[f"msg_total@{cs}s"], 1)
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Wrote:", out_path)
    for r in rows[:10]:
        print(r)


if __name__ == "__main__":
    main()
