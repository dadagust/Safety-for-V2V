from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml

from watermark.attacks import (
    aac_roundtrip_np,
    add_noise_snr_np,
    bitdepth_reduce_np,
    crop_pad_np,
    gain_np,
    lowpass_np,
    mp3_roundtrip_np,
    resample_speed_np,
)
from watermark.data import load_audio
from watermark.model import Encoder1D, Detector1D, WatermarkNet


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(cfg: Dict, checkpoint_path: str, device: torch.device) -> WatermarkNet:
    enc = Encoder1D(
        nbits=int(cfg["model"]["nbits"]),
        msg_dim=int(cfg["model"]["msg_dim"]),
        hidden=int(cfg["model"]["enc_hidden"]),
        max_delta=float(cfg["model"]["max_delta"]),
    )
    det = Detector1D(
        nbits=int(cfg["model"]["nbits"]),
        channels=list(cfg["model"]["det_channels"]),
        kernel_size=int(cfg["model"]["det_kernel_size"]),
        strides=list(cfg["model"]["det_strides"]),
    )
    model = WatermarkNet(enc, det).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def read_manifest_paths(manifest_csv: str) -> List[str]:
    import csv
    with open(manifest_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row["path"] for row in r]


def detect_and_decode(
    detector: Detector1D,
    audio_t: torch.Tensor,
    presence_threshold: float,
    min_positive_fraction: float,
) -> Tuple[bool, torch.Tensor, float]:
    """
    audio_t: [1,1,T]
    Returns: (present?, decoded_bits_logits_avg [nbits], presence_score)
    """
    with torch.no_grad():
        presence_logits, bit_logits = detector(audio_t)  # [1,T'], [1,nbits,T']
        prob = torch.sigmoid(presence_logits)[0]  # [T']
        present = (prob > presence_threshold).float().mean().item() > min_positive_fraction
        presence_score = float(prob.max().item())

        if present:
            w = (prob > presence_threshold).float()
            if w.sum().item() < 1.0:
                # fallback: use top-k frames
                k = max(1, int(0.1 * w.numel()))
                topk = torch.topk(prob, k).indices
                w = torch.zeros_like(prob)
                w[topk] = 1.0
            avg = (bit_logits[0] * w.unsqueeze(0)).sum(dim=-1) / w.sum().clamp_min(1.0)  # [nbits]
        else:
            avg = bit_logits[0].mean(dim=-1)  # [nbits], arbitrary
        return present, avg, presence_score


def build_attack_cases(cfg: Dict, sr: int) -> List[Tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    cases: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
    cases.append(("clean", lambda x: x))

    # noise
    for snr in cfg["attacks_eval"]["noise_snr_db_list"]:
        cases.append((f"noise_snr{snr}", lambda x, snr=snr: add_noise_snr_np(x, float(snr))))

    # speed
    for f in cfg["attacks_eval"]["resample_factor_list"]:
        cases.append((f"speed_{f}", lambda x, f=f: resample_speed_np(x, float(f), sr)))

    # lowpass
    for c in cfg["attacks_eval"]["lowpass_hz_list"]:
        cases.append((f"lowpass_{c}", lambda x, c=c: lowpass_np(x, sr, float(c))))

    # crop
    for cs in cfg["attacks_eval"]["crop_seconds_list"]:
        cases.append((f"crop_{cs}s", lambda x, cs=cs: crop_pad_np(x, float(cs), sr)))

    # bitdepth
    for b in cfg["attacks_eval"]["bitdepth_list"]:
        cases.append((f"bitdepth_{b}", lambda x, b=b: bitdepth_reduce_np(x, int(b))))

    # codecs
    for br in cfg["attacks_eval"]["mp3_bitrates_k"]:
        cases.append((f"mp3_{br}k", lambda x, br=br: mp3_roundtrip_np(x, sr, int(br))))
    for br in cfg["attacks_eval"]["aac_bitrates_k"]:
        cases.append((f"aac_{br}k", lambda x, br=br: aac_roundtrip_np(x, sr, int(br))))

    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max_items", type=int, default=200, help="Limit items for speed (esp. mp3/aac). 0 = all.")
    ap.add_argument("--latency", action="store_true", help="Also compute detection/msg accuracy vs chunk duration.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = int(cfg["audio"]["sample_rate"])
    seg_len = int(round(cfg["audio"]["segment_seconds"] * sr))

    model = load_model(cfg, args.checkpoint, device)
    presence_thr = float(cfg["detection"]["presence_threshold"])
    min_frac = float(cfg["detection"]["min_positive_fraction"])

    paths = read_manifest_paths(args.manifest)
    if args.max_items and args.max_items > 0:
        paths = paths[: args.max_items]

    attack_cases = build_attack_cases(cfg, sr)
    chunk_secs = list(cfg["latency_probe"]["chunk_seconds"])

    # accumulators per attack
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

        # tensors
        x = torch.from_numpy(x_np).to(device).view(1, 1, -1)

        bits = torch.randint(0, 2, (1, nbits), device=device, dtype=torch.float32)

        # watermark entire segment
        with torch.no_grad():
            y, _ = model.encoder(x, bits)
        y_np = y.detach().cpu().numpy().reshape(-1).astype(np.float32)

        # negative example: original audio
        x0_np = x_np

        for name, fn in attack_cases:
            try:
                y_att_pos = fn(y_np)
                y_att_neg = fn(x0_np)
            except subprocess.CalledProcessError:
                # ffmpeg failed; skip this case for this file
                continue

            # run detector
            y_pos_t = torch.from_numpy(y_att_pos).to(device).view(1, 1, -1)
            y_neg_t = torch.from_numpy(y_att_neg).to(device).view(1, 1, -1)

            pos_present, pos_bits_logits, _ = detect_and_decode(model.detector, y_pos_t, presence_thr, min_frac)
            neg_present, _neg_bits_logits, _ = detect_and_decode(model.detector, y_neg_t, presence_thr, min_frac)

            # update confusion
            if pos_present:
                stats[name]["tp"] += 1
            else:
                stats[name]["fn"] += 1

            if neg_present:
                stats[name]["fp"] += 1
            else:
                stats[name]["tn"] += 1

            # bits only if detected positive
            if pos_present:
                pred_bits = (pos_bits_logits > 0).float()
                stats[name]["bit_correct"] += int((pred_bits == bits[0]).sum().item())
                stats[name]["bit_total"] += nbits
                stats[name]["msg_correct"] += int(bool(torch.all(pred_bits == bits[0]).item()))
                stats[name]["msg_total"] += 1

            # latency probe
            if args.latency:
                for cs in chunk_secs:
                    Tchunk = max(64, int(round(float(cs) * sr)))
                    y_chunk = y_att_pos[:Tchunk]
                    y_chunk_t = torch.from_numpy(y_chunk).to(device).view(1, 1, -1)
                    present_c, bits_logits_c, _ = detect_and_decode(model.detector, y_chunk_t, presence_thr, min_frac)
                    if present_c:
                        stats[name][f"tp@{cs}s"] += 1
                        pred_bits_c = (bits_logits_c > 0).float()
                        stats[name][f"msg_correct@{cs}s"] += int(bool(torch.all(pred_bits_c == bits[0]).item()))
                    stats[name][f"msg_total@{cs}s"] += 1

    # write CSV
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
                row[f"TPR@{cs}s"] = s[f"tp@{cs}s"] / max(s["tp"] + s["fn"], 1)  # relative to total positives attempted
                row[f"msg_acc@{cs}s"] = s[f"msg_correct@{cs}s"] / max(s[f"msg_total@{cs}s"], 1)
        rows.append(row)

    # stable header
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
