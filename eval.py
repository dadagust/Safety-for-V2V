from __future__ import annotations

import argparse
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from watermark.attacks import DifferentiableAttacks, TrainAttackConfig
from watermark.data import AudioManifestDataset
from watermark.runtime import (
    build_model_from_cfg,
    detector_forward_with_predicted_mask,
    downsample_mask,
    embed_batch,
    predicted_binary_mask,
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


def resolve_embed_mode(cfg: Dict[str, Any], arg_mode: str) -> str:
    if arg_mode != "auto":
        return arg_mode
    mode = str((cfg.get("embed", {}) or {}).get("mode", "window")).lower()
    if mode in ("full", "window", "dropout", "dropout_holes", "holes"):
        return "dropout" if mode in ("dropout_holes", "holes") else mode
    return "window"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--no_attacks", action="store_true", help="Disable differentiable eval attacks")
    ap.add_argument(
        "--embed_mode",
        type=str,
        default="auto",
        choices=["auto", "full", "window", "dropout"],
        help="How to embed positives for evaluation. 'auto' follows the config.",
    )
    ap.add_argument("--use_hard_mask", action="store_true", help="Decode bits with thresholded predicted mask instead of soft weights")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)

    sr = int(cfg["audio"]["sample_rate"])
    wm_len = int(round(float(cfg["audio"]["watermark_seconds"]) * sr))
    ds = AudioManifestDataset(args.manifest, sample_rate=sr, segment_seconds=cfg["audio"]["segment_seconds"], random_crop=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = load_model(cfg, args.checkpoint, device)

    attacks = None
    if not args.no_attacks:
        atk_cfg = TrainAttackConfig(**cfg["attacks_train"])
        attacks = DifferentiableAttacks(atk_cfg, sample_rate=sr)

    thr = float(cfg["detection"]["presence_threshold"])
    min_frac = float(cfg["detection"]["min_positive_fraction"])
    embed_mode = resolve_embed_mode(cfg, args.embed_mode)

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

        y_clean, mask = embed_batch(
            x=x,
            bits=bits,
            model=model,
            cfg=cfg,
            sample_rate=sr,
            watermark_len=wm_len,
            p_embed=0.5,
            embed_mode_override=embed_mode,
        )
        if attacks is None:
            y = y_clean
        else:
            y, _ = attacks(y_clean, None)

        presence_logits, bit_logits, _weights = detector_forward_with_predicted_mask(
            model.detector,
            y,
            threshold=thr,
            use_soft_mask=not args.use_hard_mask,
        )
        Tp = presence_logits.shape[-1]
        mask_ds = downsample_mask(mask, Tp)
        gt_present = (mask_ds.max(dim=-1).values > 0.5)
        pred_present = present_from_presence_logits(presence_logits, threshold=thr, min_positive_fraction=min_frac)

        for b in range(B):
            gt = bool(gt_present[b].item())
            pr = bool(pred_present[b].item())
            if gt and pr:
                tp += 1
            elif (not gt) and pr:
                fp += 1
            elif (not gt) and (not pr):
                tn += 1
            else:
                fn += 1

        if embed_mode in ("window", "dropout"):
            gt_m = (mask_ds > 0.5).float()
            pr_m = predicted_binary_mask(presence_logits, threshold=thr)
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

        pred_bits = (torch.sigmoid(bit_logits) >= 0.5).float()
        for b in range(B):
            if not bool(gt_present[b].item()):
                continue
            if not bool(pred_present[b].item()):
                continue
            bit_correct += int((pred_bits[b] == bits[b]).sum().item())
            bit_total += nbits
            msg_correct += int(bool(torch.all(pred_bits[b] == bits[b]).item()))
            msg_total += 1

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        postfix = {"TPR": f"{tpr:.3f}", "FPR": f"{fpr:.3f}", "msg_acc": f"{(msg_correct/max(msg_total,1)):.3f}"}
        if embed_mode in ("window", "dropout") and loc_n > 0:
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
    if embed_mode in ("window", "dropout"):
        if loc_n > 0:
            print(f"Localization IoU (avg over positive samples) = {loc_iou_sum/loc_n:.4f}")
            print(f"Localization F1  (avg over positive samples) = {loc_f1_sum/loc_n:.4f}")
        else:
            print("Localization metrics: no positive samples were scored (unexpected).")


if __name__ == "__main__":
    main()
