from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from watermark.attacks import DifferentiableAttacks, TrainAttackConfig
from watermark.data import AudioManifestDataset
from watermark.losses import multiscale_stft_loss, mse_loss
from watermark.runtime import build_model_from_cfg, detector_forward_with_gt_mask, embed_batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(path: str, model, opt, step: int, cfg: Dict[str, Any], epoch: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": int(step),
            "epoch": int(epoch),
            "cfg": cfg,
        },
        path,
    )


def maybe_balanced_bce(logits: torch.Tensor, targets: torch.Tensor, enabled: bool = False) -> torch.Tensor:
    if not enabled:
        return F.binary_cross_entropy_with_logits(logits, targets)

    with torch.no_grad():
        pos_frac = targets.mean().clamp(1e-4, 1.0 - 1e-4)
        pos_weight = ((1.0 - pos_frac) / pos_frac).clamp(1.0, 20.0)

    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype),
    )


def compute_losses(
    x: torch.Tensor,
    y_clean: torch.Tensor,
    y_attacked: torch.Tensor,
    mask: torch.Tensor,
    bits: torch.Tensor,
    detector,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    presence_logits, bit_logits, mask_ds = detector_forward_with_gt_mask(detector, y_attacked, mask)

    presence_loss = maybe_balanced_bce(
        presence_logits,
        mask_ds,
        enabled=bool(cfg.get("loss", {}).get("balance_presence", False)),
    )

    wsum = mask_ds.sum(dim=-1)
    has_pos = (wsum > 1e-4).float()
    bit_bce = F.binary_cross_entropy_with_logits(bit_logits, bits, reduction="none").mean(dim=1)
    bit_loss = (bit_bce * has_pos).sum() / has_pos.sum().clamp_min(1.0)

    l2 = mse_loss(x, y_clean)
    stft = multiscale_stft_loss(x, y_clean, cfg["loss"]["stft_scales"])

    total = (
        float(cfg["loss"]["presence_weight"]) * presence_loss
        + float(cfg["loss"]["bit_weight"]) * bit_loss
        + float(cfg["loss"]["l2_weight"]) * l2
        + float(cfg["loss"]["stft_weight"]) * stft
    )

    with torch.no_grad():
        bit_pred = (torch.sigmoid(bit_logits) >= 0.5).float()
        bit_acc_per_sample = (bit_pred == bits).float().mean(dim=1)
        bit_acc = (bit_acc_per_sample * has_pos).sum() / has_pos.sum().clamp_min(1.0)

        pr_mask = (torch.sigmoid(presence_logits) >= float(cfg.get("detection", {}).get("presence_threshold", 0.5))).float()
        gt_mask = (mask_ds >= 0.5).float()
        inter = (pr_mask * gt_mask).sum(dim=-1)
        union = ((pr_mask + gt_mask) > 0).float().sum(dim=-1).clamp_min(1.0)
        frame_iou = (inter / union).mean()

    logs = {
        "loss_total": float(total.item()),
        "loss_presence": float(presence_loss.item()),
        "loss_bits": float(bit_loss.item()),
        "loss_l2": float(l2.item()),
        "loss_stft": float(stft.item()),
        "bit_acc": float(bit_acc.item()),
        "frame_iou": float(frame_iou.item()),
        "wm_frac": float(mask.mean().item()),
    }
    return total, logs


def compute_detection_losses_only(
    y_attacked: torch.Tensor,
    mask: torch.Tensor,
    bits: torch.Tensor,
    detector,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    presence_logits, bit_logits, mask_ds = detector_forward_with_gt_mask(detector, y_attacked, mask)

    presence_loss = maybe_balanced_bce(
        presence_logits,
        mask_ds,
        enabled=bool(cfg.get("loss", {}).get("balance_presence", False)),
    )

    wsum = mask_ds.sum(dim=-1)
    has_pos = (wsum > 1e-4).float()
    bit_bce = F.binary_cross_entropy_with_logits(bit_logits, bits, reduction="none").mean(dim=1)
    bit_loss = (bit_bce * has_pos).sum() / has_pos.sum().clamp_min(1.0)
    return presence_loss, bit_loss


@torch.no_grad()
def run_validation(
    model,
    val_dl: DataLoader,
    attacks: DifferentiableAttacks | None,
    cfg: Dict[str, Any],
    sr: int,
    wm_len: int,
    device: torch.device,
    attacked: bool,
) -> Dict[str, float]:
    model.eval()
    agg = {
        "loss_total": [],
        "loss_presence": [],
        "loss_bits": [],
        "loss_l2": [],
        "loss_stft": [],
        "bit_acc": [],
        "frame_iou": [],
        "wm_frac": [],
    }

    for batch in val_dl:
        x = batch["audio"].to(device, non_blocking=True)
        B = x.shape[0]
        bits = torch.randint(0, 2, (B, int(cfg["model"]["nbits"])), device=device, dtype=torch.float32)

        y_clean, mask = embed_batch(
            x=x,
            bits=bits,
            model=model,
            cfg=cfg,
            sample_rate=sr,
            watermark_len=wm_len,
            p_embed=1.0,
        )

        if attacked and attacks is not None:
            y_eval, mask_eval = attacks(y_clean, mask)
            mask_use = mask_eval if mask_eval is not None else mask
        else:
            y_eval, mask_use = y_clean, mask

        _, logs = compute_losses(
            x=x,
            y_clean=y_clean,
            y_attacked=y_eval,
            mask=mask_use,
            bits=bits,
            detector=model.detector,
            cfg=cfg,
        )

        for k in agg.keys():
            agg[k].append(logs[k])

    return {k: float(np.mean(v)) if len(v) else 0.0 for k, v in agg.items()}

def sample_bits(B: int, cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    nbits = int(cfg["model"]["nbits"])
    train_cfg = cfg.get("train", {}) or {}

    fixed_bits = train_cfg.get("fixed_bits", None)
    codebook = train_cfg.get("codebook", None)

    if fixed_bits is not None:
        fb = torch.tensor([fixed_bits], device=device, dtype=torch.float32)
        assert fb.shape[1] == nbits
        return fb.repeat(B, 1)

    if codebook is not None:
        cb = torch.tensor(codebook, device=device, dtype=torch.float32)  # [K, nbits]
        idx = torch.randint(0, cb.shape[0], (B,), device=device)
        return cb[idx]

    return torch.randint(0, 2, (B, nbits), device=device, dtype=torch.float32)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--train_manifest", type=str, required=True)
    ap.add_argument("--val_manifest", type=str, required=True)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument(
        "--resume_optimizer",
        action="store_true",
        help="Also resume optimizer state. By default only model weights are loaded.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("CONFIG PATH:", args.config)
    print(yaml.safe_dump(cfg, sort_keys=False))
    print("bit_weight =", cfg["loss"]["bit_weight"])
    print("p_watermark =", cfg["train"]["p_watermark"])
    print("max_delta =", cfg["model"]["max_delta"])

    set_seed(int(cfg.get("seed", 1337)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    print("Device:", device)

    use_amp = bool(cfg.get("train", {}).get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print("AMP:", use_amp)

    sr = int(cfg["audio"]["sample_rate"])
    wm_len = int(round(float(cfg["audio"]["watermark_seconds"]) * sr))

    latency_cfg = cfg.get("latency_train", {}) or {}
    latency_enable = bool(latency_cfg.get("enable", False))
    latency_weight = float(latency_cfg.get("weight", 0.5))
    latency_p = float(latency_cfg.get("p_apply", 0.5))
    latency_chunks = latency_cfg.get("chunk_seconds", cfg.get("latency_probe", {}).get("chunk_seconds", [0.5]))
    latency_chunks = [float(x) for x in latency_chunks]

    train_ds = AudioManifestDataset(
        args.train_manifest,
        sample_rate=sr,
        segment_seconds=float(cfg["audio"]["segment_seconds"]),
        random_crop=True,
    )
    val_ds = AudioManifestDataset(
        args.val_manifest,
        sample_rate=sr,
        segment_seconds=float(cfg["audio"]["segment_seconds"]),
        random_crop=False,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model_from_cfg(cfg, device=device)
    if hasattr(model.encoder, "receptive_field"):
        print("Encoder receptive field (samples):", model.encoder.receptive_field())
    print("Detector downsample factor:", getattr(model.detector, "downsample_factor", "?"))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
        betas=(0.9, 0.999),
    )

    start_step = 0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        start_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Loaded model weights from {args.resume} step={start_step} epoch={start_epoch}")

        if args.resume_optimizer:
            opt.load_state_dict(ckpt["opt"])
            for g in opt.param_groups:
                g["lr"] = float(cfg["train"]["lr"])
            print("Optimizer state resumed and LR overwritten from current config.")
        else:
            print("Optimizer state NOT resumed (recommended for stage transitions).")

    atk_cfg = TrainAttackConfig(**cfg["attacks_train"])
    train_attacks = DifferentiableAttacks(atk_cfg, sample_rate=sr)

    steps_per_epoch = int(cfg["train"]["steps_per_epoch"])
    save_every = int(cfg["train"]["save_every_steps"])
    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = start_step
    best_clean_bit_acc = -1.0

    for epoch in range(start_epoch, start_epoch + int(cfg["train"]["epochs"])):
        model.train()
        pbar = tqdm(total=steps_per_epoch, desc=f"epoch {epoch + 1}")
        dl_iter = iter(train_dl)

        for _ in range(steps_per_epoch):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(train_dl)
                batch = next(dl_iter)

            x = batch["audio"].to(device, non_blocking=True)
            B = x.shape[0]
            bits = sample_bits(B, cfg, device)
            y_clean, mask = embed_batch(
                x=x,
                bits=bits,
                model=model,
                cfg=cfg,
                sample_rate=sr,
                watermark_len=wm_len,
                p_embed=float(cfg["train"]["p_watermark"]),
            )

            y_att, mask_att = train_attacks(y_clean, mask)
            mask_use = mask_att if mask_att is not None else mask

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, logs = compute_losses(
                    x=x,
                    y_clean=y_clean,
                    y_attacked=y_att,
                    mask=mask_use,
                    bits=bits,
                    detector=model.detector,
                    cfg=cfg,
                )

                if latency_enable and (torch.rand((), device=device).item() < latency_p):
                    cs = float(latency_chunks[int(torch.randint(0, len(latency_chunks), (1,), device=device).item())])
                    Tchunk = max(64, int(round(cs * sr)))
                    y_chunk = y_att[:, :, :Tchunk]
                    m_chunk = mask_use[:, :, :Tchunk]
                    pres_l, bit_l = compute_detection_losses_only(y_chunk, m_chunk, bits, model.detector, cfg)
                    lat_loss = float(cfg["loss"]["presence_weight"]) * pres_l + float(cfg["loss"]["bit_weight"]) * bit_l
                    loss = loss + latency_weight * lat_loss
                    logs["loss_latency"] = float(lat_loss.item())

            scaler.scale(loss).backward()

            if float(cfg["train"].get("grad_clip", 0.0)) > 0:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                logs["grad_norm"] = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)

            scaler.step(opt)
            scaler.update()

            global_step += 1
            pbar.update(1)

            if global_step % int(cfg["train"]["log_every"]) == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})

            if global_step % save_every == 0:
                save_checkpoint(str(ckpt_dir / "latest.pt"), model, opt, global_step, cfg, epoch + 1)
                save_checkpoint(str(ckpt_dir / f"step_{global_step}.pt"), model, opt, global_step, cfg, epoch + 1)

        pbar.close()

        val_clean = run_validation(
            model=model,
            val_dl=val_dl,
            attacks=None,
            cfg=cfg,
            sr=sr,
            wm_len=wm_len,
            device=device,
            attacked=False,
        )
        print(
            f"[val-clean] epoch {epoch+1} "
            f"loss_total={val_clean['loss_total']:.4f} "
            f"presence={val_clean['loss_presence']:.4f} "
            f"bits={val_clean['loss_bits']:.4f} "
            f"bit_acc={val_clean['bit_acc']:.4f} "
            f"iou={val_clean['frame_iou']:.4f} "
            f"wm_frac={val_clean['wm_frac']:.4f}"
        )

        val_att = run_validation(
            model=model,
            val_dl=val_dl,
            attacks=train_attacks,
            cfg=cfg,
            sr=sr,
            wm_len=wm_len,
            device=device,
            attacked=True,
        )
        print(
            f"[val-attacked] epoch {epoch+1} "
            f"loss_total={val_att['loss_total']:.4f} "
            f"presence={val_att['loss_presence']:.4f} "
            f"bits={val_att['loss_bits']:.4f} "
            f"bit_acc={val_att['bit_acc']:.4f} "
            f"iou={val_att['frame_iou']:.4f} "
            f"wm_frac={val_att['wm_frac']:.4f}"
        )

        save_checkpoint(str(ckpt_dir / "latest.pt"), model, opt, global_step, cfg, epoch + 1)

        if val_clean["bit_acc"] > best_clean_bit_acc:
            best_clean_bit_acc = val_clean["bit_acc"]
            save_checkpoint(str(ckpt_dir / "best_clean_bitacc.pt"), model, opt, global_step, cfg, epoch + 1)
            print(f"Saved best_clean_bitacc.pt with clean bit_acc={best_clean_bit_acc:.4f}")

    print("Done. Latest checkpoint:", ckpt_dir / "latest.pt")


if __name__ == "__main__":
    main()
