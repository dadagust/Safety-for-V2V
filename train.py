from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
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
from watermark.losses import multiscale_stft_loss, mse_loss
from watermark.embed import embed_random_window, embed_dropout_holes
from watermark.model import Encoder1D, Detector1D, WatermarkNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(path: str, model: WatermarkNet, opt: torch.optim.Optimizer, step: int, cfg: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "cfg": cfg,
        },
        path,
    )


@torch.no_grad()
def _downsample_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    # mask: [B,1,T] -> [B, target_len]
    m = F.interpolate(mask, size=target_len, mode="linear", align_corners=False)
    return m.squeeze(1).clamp(0.0, 1.0)


def compute_losses(
    x: torch.Tensor,
    y_clean: torch.Tensor,
    y_attacked: torch.Tensor,
    mask: torch.Tensor,
    bits: torch.Tensor,
    detector: Detector1D,
    cfg: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns total loss and scalar logs.
    """
    presence_logits, bit_logits = detector(y_attacked)  # [B,T'], [B,nbits,T']
    B, nbits, Tp = bit_logits.shape

    mask_ds = _downsample_mask(mask, Tp)  # [B,T']

    # Presence loss (frame-level)
    presence_loss = F.binary_cross_entropy_with_logits(presence_logits, mask_ds)

    # Bit loss: pool only frames with watermark
    weights = mask_ds  # [B,T']
    wsum = weights.sum(dim=-1)  # [B]
    has_pos = (wsum > 1e-4).float()

    # weighted average logits across time
    bit_avg = (bit_logits * weights.unsqueeze(1)).sum(dim=-1) / (wsum.unsqueeze(1).clamp_min(1e-4))
    bit_bce = F.binary_cross_entropy_with_logits(bit_avg, bits, reduction="none").mean(dim=1)  # [B]
    bit_loss = (bit_bce * has_pos).sum() / has_pos.sum().clamp_min(1.0)

    # Imperceptibility losses (compare pre-attack watermarked audio to original)
    l2 = mse_loss(x, y_clean)
    stft = multiscale_stft_loss(x, y_clean, cfg["loss"]["stft_scales"])

    total = (
        cfg["loss"]["presence_weight"] * presence_loss
        + cfg["loss"]["bit_weight"] * bit_loss
        + cfg["loss"]["l2_weight"] * l2
        + cfg["loss"]["stft_weight"] * stft
    )

    logs = {
        "loss_total": float(total.item()),
        "loss_presence": float(presence_loss.item()),
        "loss_bits": float(bit_loss.item()),
        "loss_l2": float(l2.item()),
        "loss_stft": float(stft.item()),
        "wm_frac": float(mask.mean().item()),
    }
    return total, logs


def compute_detection_losses_only(
    y_attacked: torch.Tensor,
    mask: torch.Tensor,
    bits: torch.Tensor,
    detector: Detector1D,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Presence + bit losses only (no imperceptibility terms).

    Useful for low-latency auxiliary training on short chunks.
    """
    presence_logits, bit_logits = detector(y_attacked)  # [B,T'], [B,nbits,T']
    B, nbits, Tp = bit_logits.shape
    mask_ds = _downsample_mask(mask, Tp)  # [B,T']

    presence_loss = F.binary_cross_entropy_with_logits(presence_logits, mask_ds)

    weights = mask_ds
    wsum = weights.sum(dim=-1)
    has_pos = (wsum > 1e-4).float()
    bit_avg = (bit_logits * weights.unsqueeze(1)).sum(dim=-1) / (wsum.unsqueeze(1).clamp_min(1e-4))
    bit_bce = F.binary_cross_entropy_with_logits(bit_avg, bits, reduction="none").mean(dim=1)
    bit_loss = (bit_bce * has_pos).sum() / has_pos.sum().clamp_min(1.0)

    return presence_loss, bit_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--train_manifest", type=str, required=True)
    ap.add_argument("--val_manifest", type=str, required=True)
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("CONFIG PATH:", args.config)
    print(yaml.safe_dump(cfg, sort_keys=False))
    print("bit_weight =", cfg["loss"]["bit_weight"])
    print("p_watermark =", cfg["train"]["p_watermark"])
    print("max_delta =", cfg["model"]["max_delta"])
    set_seed(int(cfg.get("seed", 1337)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    sr = int(cfg["audio"]["sample_rate"])
    seg_len = int(round(cfg["audio"]["segment_seconds"] * sr))
    wm_len = int(round(cfg["audio"]["watermark_seconds"] * sr))

    # Embedding / Variant selection
    embed_cfg = cfg.get("embed", {}) or {}
    embed_mode = str(embed_cfg.get("mode", "random_window")).lower()
    window_start_mode = str(embed_cfg.get("start_mode", "random")).lower()

    # Optional low-latency auxiliary training
    latency_cfg = cfg.get("latency_train", {}) or {}
    latency_enable = bool(latency_cfg.get("enable", False))
    latency_weight = float(latency_cfg.get("weight", 0.5))
    latency_p = float(latency_cfg.get("p_apply", 0.5))
    latency_chunks = latency_cfg.get("chunk_seconds", cfg.get("latency_probe", {}).get("chunk_seconds", [0.5]))
    latency_chunks = [float(x) for x in latency_chunks]

    train_ds = AudioManifestDataset(args.train_manifest, sample_rate=sr, segment_seconds=cfg["audio"]["segment_seconds"], random_crop=True)
    val_ds = AudioManifestDataset(args.val_manifest, sample_rate=sr, segment_seconds=cfg["audio"]["segment_seconds"], random_crop=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        drop_last=False,
    )

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

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = int(ckpt.get("step", 0))
        print(f"Resumed from {args.resume} step={start_step}")

    atk_cfg = TrainAttackConfig(**cfg["attacks_train"])
    attacks = DifferentiableAttacks(atk_cfg, sample_rate=sr)

    steps_per_epoch = int(cfg["train"]["steps_per_epoch"])
    save_every = int(cfg["train"]["save_every_steps"])
    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = start_step
    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        pbar = tqdm(total=steps_per_epoch, desc=f"epoch {epoch+1}")
        dl_iter = iter(train_dl)

        for _ in range(steps_per_epoch):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(train_dl)
                batch = next(dl_iter)

            x = batch["audio"].to(device)  # [B,1,T]
            B = x.shape[0]

            bits = torch.randint(0, 2, (B, int(cfg["model"]["nbits"])), device=device, dtype=torch.float32)

            if embed_mode in ("dropout", "dropout_holes", "holes"):
                nh = embed_cfg.get("num_holes", (0, 3))
                hs = embed_cfg.get("hole_seconds", (0.25, 1.0))
                y_clean, mask = embed_dropout_holes(
                    x=x,
                    bits=bits,
                    encoder=model.encoder,
                    sample_rate=sr,
                    p_embed=float(cfg["train"]["p_watermark"]),
                    ensure_prefix_seconds=float(embed_cfg.get("ensure_prefix_seconds", 0.5)),
                    num_holes=(int(nh[0]), int(nh[1])),
                    hole_seconds=(float(hs[0]), float(hs[1])),
                    ramp_ms=float(embed_cfg.get("ramp_ms", 10.0)),
                )
            else:
                y_clean, mask = embed_random_window(
                    x=x,
                    bits=bits,
                    encoder=model.encoder,
                    watermark_len=wm_len,
                    p_embed=float(cfg["train"]["p_watermark"]),
                    start_mode=window_start_mode,
                )

            y_att, mask_att = attacks(y_clean, mask)

            loss, logs = compute_losses(
                x=x,
                y_clean=y_clean,
                y_attacked=y_att,
                mask=mask_att if mask_att is not None else mask,
                bits=bits,
                detector=model.detector,
                cfg=cfg,
            )

            # Optional: encourage detection/decoding from short chunks (low latency)
            if latency_enable and (torch.rand((), device=device).item() < latency_p):
                cs = float(latency_chunks[int(torch.randint(0, len(latency_chunks), (1,), device=device).item())])
                Tchunk = max(64, int(round(cs * sr)))
                y_chunk = y_att[:, :, :Tchunk]
                m_chunk = (mask_att if mask_att is not None else mask)[:, :, :Tchunk]
                pres_l, bit_l = compute_detection_losses_only(y_chunk, m_chunk, bits, model.detector)
                lat_loss = float(cfg["loss"]["presence_weight"]) * pres_l + float(cfg["loss"]["bit_weight"]) * bit_l
                loss = loss + latency_weight * lat_loss
                logs["loss_latency"] = float(lat_loss.item())

            opt.zero_grad(set_to_none=True)
            loss.backward()

            # grad clip
            if float(cfg["train"].get("grad_clip", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))

            opt.step()

            global_step += 1
            pbar.update(1)
            if global_step % int(cfg["train"]["log_every"]) == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})

            if global_step % save_every == 0:
                save_checkpoint(str(ckpt_dir / "latest.pt"), model, opt, global_step, cfg)
                save_checkpoint(str(ckpt_dir / f"step_{global_step}.pt"), model, opt, global_step, cfg)

        pbar.close()

        # quick val
        model.eval()
        with torch.no_grad():
            vals = []
            for batch in val_dl:
                x = batch["audio"].to(device)
                B = x.shape[0]
                bits = torch.randint(0, 2, (B, int(cfg["model"]["nbits"])), device=device, dtype=torch.float32)
                if embed_mode in ("dropout", "dropout_holes", "holes"):
                    nh = embed_cfg.get("num_holes", (0, 3))
                    hs = embed_cfg.get("hole_seconds", (0.25, 1.0))
                    y_clean, mask = embed_dropout_holes(
                        x=x,
                        bits=bits,
                        encoder=model.encoder,
                        sample_rate=sr,
                        p_embed=1.0,
                        ensure_prefix_seconds=float(embed_cfg.get("ensure_prefix_seconds", 0.5)),
                        num_holes=(int(nh[0]), int(nh[1])),
                        hole_seconds=(float(hs[0]), float(hs[1])),
                        ramp_ms=float(embed_cfg.get("ramp_ms", 10.0)),
                    )
                else:
                    y_clean, mask = embed_random_window(
                        x=x,
                        bits=bits,
                        encoder=model.encoder,
                        watermark_len=wm_len,
                        p_embed=1.0,
                        start_mode=window_start_mode,
                    )
                y_att, mask_att = attacks(y_clean, mask)
                loss, logs = compute_losses(x, y_clean, y_att, mask_att if mask_att is not None else mask, bits, model.detector, cfg)
                vals.append(logs["loss_total"])
            print(f"[val] epoch {epoch+1} loss_total={float(np.mean(vals)):.4f}")

        save_checkpoint(str(ckpt_dir / "latest.pt"), model, opt, global_step, cfg)

    print("Done. Latest checkpoint:", ckpt_dir / "latest.pt")


if __name__ == "__main__":
    main()
