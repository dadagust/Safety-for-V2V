from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from watermark.attacks import DifferentiableAttacks, TrainAttackConfig
from watermark.data import AudioManifestDataset
from watermark.embed import embed_full, embed_random_window, embed_dropout_holes
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


@torch.no_grad()
def downsample_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    m = F.interpolate(mask, size=target_len, mode="linear", align_corners=False)
    return m.squeeze(1).clamp(0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--no_attacks", action="store_true", help="Disable differentiable eval attacks")
    ap.add_argument(
        "--embed_mode",
        type=str,
        default="full",
        choices=["full", "window", "dropout"],
        help="How to embed positives for evaluation. 'dropout' produces a localized presence mask.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sr = int(cfg["audio"]["sample_rate"])
    ds = AudioManifestDataset(args.manifest, sample_rate=sr, segment_seconds=cfg["audio"]["segment_seconds"], random_crop=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = load_model(cfg, args.checkpoint, device)

    attacks = None
    if not args.no_attacks:
        atk_cfg = TrainAttackConfig(**cfg["attacks_train"])
        attacks = DifferentiableAttacks(atk_cfg, sample_rate=sr)

    thr = float(cfg["detection"]["presence_threshold"])
    min_frac = float(cfg["detection"]["min_positive_fraction"])

    # metrics
    tp = fp = tn = fn = 0
    bit_correct = 0
    bit_total = 0
    msg_correct = 0
    msg_total = 0
    loc_iou_sum = 0.0
    loc_f1_sum = 0.0
    loc_n = 0

    pbar = tqdm(dl, desc="eval")
    for batch in pbar:
        x = batch["audio"].to(device)
        B = x.shape[0]
        nbits = int(cfg["model"]["nbits"])

        bits = torch.randint(0, 2, (B, nbits), device=device, dtype=torch.float32)

        # half positive / half negative
        do_embed = torch.rand((B,), device=device) < 0.5

        if args.embed_mode == "full":
            y_clean, mask = embed_full(x, bits, model.encoder, do_embed)
        elif args.embed_mode == "window":
            # embed a single window; choose 2s by default (from config) and start randomly
            wm_len = int(round(float(cfg["audio"].get("watermark_seconds", 2.0)) * sr))
            y_clean, mask = embed_random_window(
                x=x,
                bits=bits,
                encoder=model.encoder,
                watermark_len=wm_len,
                p_embed=0.5,
                start_mode=str((cfg.get("embed", {}) or {}).get("start_mode", "random")),
            )
        else:  # dropout
            e = cfg.get("embed", {}) or {}
            nh = e.get("num_holes", (0, 3))
            hs = e.get("hole_seconds", (0.25, 1.0))
            y_clean, mask = embed_dropout_holes(
                x=x,
                bits=bits,
                encoder=model.encoder,
                sample_rate=sr,
                p_embed=0.5,
                ensure_prefix_seconds=float(e.get("ensure_prefix_seconds", 0.5)),
                num_holes=(int(nh[0]), int(nh[1])),
                hole_seconds=(float(hs[0]), float(hs[1])),
                ramp_ms=float(e.get("ramp_ms", 10.0)),
            )
        if attacks is None:
            y = y_clean
        else:
            y, _ = attacks(y_clean, None)

        presence_logits, bit_logits = model.detector(y)
        Tp = presence_logits.shape[1]
        mask_ds = downsample_mask(mask, Tp)  # [B,T']
        # present if *any* watermark frames exist
        gt_present = (mask_ds.max(dim=-1).values > 0.5)  # [B] bool

        prob = torch.sigmoid(presence_logits)  # [B,T']
        pred_present = (prob > thr).float().mean(dim=-1) > min_frac  # [B] bool

        # confusion matrix
        for b in range(B):
            gt = bool(gt_present[b].item())
            pr = bool(pred_present[b].item())
            if gt and pr:
                tp += 1
            elif (not gt) and pr:
                fp += 1
            elif (not gt) and (not pr):
                tn += 1
            elif gt and (not pr):
                fn += 1

        # localization metrics (only meaningful when mask is not full/constant)
        if args.embed_mode in ("window", "dropout"):
            # frame-level binary masks
            gt_m = (mask_ds > 0.5).float()
            pr_m = (prob > thr).float()
            # only score samples that contain watermark in GT
            for b in range(B):
                if gt_m[b].sum().item() < 1.0:
                    continue
                inter = (gt_m[b] * pr_m[b]).sum().item()
                union = (gt_m[b] + pr_m[b] - gt_m[b] * pr_m[b]).sum().item()
                gt_sum = gt_m[b].sum().item()
                pr_sum = pr_m[b].sum().item()
                fp_f = pr_sum - inter
                fn_f = gt_sum - inter
                iou = inter / max(union, 1e-8)
                f1 = (2.0 * inter) / max(2.0 * inter + fp_f + fn_f, 1e-8)
                loc_iou_sum += float(iou)
                loc_f1_sum += float(f1)
                loc_n += 1

        # bit metrics on positives that were detected
        for b in range(B):
            if not bool(gt_present[b].item()):
                continue
            if not bool(pred_present[b].item()):
                continue

            w = (prob[b] > thr).float()  # [T']
            if w.sum().item() < 1.0:
                w = mask_ds[b]  # fallback
            avg = (bit_logits[b] * w.unsqueeze(0)).sum(dim=-1) / w.sum().clamp_min(1.0)
            pred_bits = (avg > 0).float()

            bit_correct += int((pred_bits == bits[b]).sum().item())
            bit_total += nbits
            msg_correct += int(bool(torch.all(pred_bits == bits[b]).item()))
            msg_total += 1

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        postfix = {"TPR": f"{tpr:.3f}", "FPR": f"{fpr:.3f}", "msg_acc": f"{(msg_correct/max(msg_total,1)):.3f}"}
        if args.embed_mode in ("window", "dropout") and loc_n > 0:
            postfix["loc_iou"] = f"{(loc_iou_sum/loc_n):.3f}"
            postfix["loc_f1"] = f"{(loc_f1_sum/loc_n):.3f}"
        pbar.set_postfix(postfix)

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    bit_acc = bit_correct / max(bit_total, 1)
    msg_acc = msg_correct / max(msg_total, 1)

    print("---- Results ----")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"TPR={tpr:.4f} FPR={fpr:.4f}")
    print(f"Bit accuracy (detected positives) = {bit_acc:.4f}")
    print(f"Message accuracy (all bits correct) = {msg_acc:.4f}")
    if args.embed_mode in ("window", "dropout"):
        if loc_n > 0:
            print(f"Localization IoU (avg over positive samples) = {loc_iou_sum/loc_n:.4f}")
            print(f"Localization F1  (avg over positive samples) = {loc_f1_sum/loc_n:.4f}")
        else:
            print("Localization metrics: no positive samples were scored (unexpected).")


if __name__ == "__main__":
    main()
