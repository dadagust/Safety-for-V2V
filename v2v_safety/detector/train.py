from __future__ import annotations
import os, json, random
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from v2v_safety.detector.data import AudioCSVDataset, AugmentConfig, AugmentConfig as AC
from v2v_safety.detector.models.cnn import SimpleSpecCNN

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)

def train_detector(cfg: Dict[str, Any], train_csv: str, val_csv: str, out_dir: str, num_classes: int = 2):
    os.makedirs(out_dir, exist_ok=True)
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    audio_cfg = cfg["audio"]
    feat_cfg = cfg["features"]
    tr_cfg = cfg["train"]
    aug_cfg = cfg.get("augment", {})

    augment = AugmentConfig(
        enable=bool(aug_cfg.get("enable", True)),
        noise_prob=float(aug_cfg.get("noise_prob", 0.5)),
        noise_snr_db_min=float(aug_cfg.get("noise_snr_db_min", 10)),
        noise_snr_db_max=float(aug_cfg.get("noise_snr_db_max", 30)),
        reverb_prob=float(aug_cfg.get("reverb_prob", 0.2)),
        rir_dir=aug_cfg.get("rir_dir", None),
        codec_prob=float(aug_cfg.get("codec_prob", 0.0)),
        codec=str(aug_cfg.get("codec", "opus")),
        codec_bitrate=int(aug_cfg.get("codec_bitrate", 16000)),
    )

    train_ds = AudioCSVDataset(
        train_csv,
        sr=int(audio_cfg["sr"]),
        clip_seconds=float(audio_cfg["clip_seconds"]),
        vad=bool(audio_cfg.get("vad", False)),
        augment=augment,
        features=True,
        n_mels=int(feat_cfg["n_mels"]),
        n_fft=int(feat_cfg["n_fft"]),
        hop_length=int(feat_cfg["hop_length"]),
        win_length=int(feat_cfg["win_length"]),
        num_classes=num_classes,
    )
    val_ds = AudioCSVDataset(
        val_csv,
        sr=int(audio_cfg["sr"]),
        clip_seconds=float(audio_cfg["clip_seconds"]),
        vad=bool(audio_cfg.get("vad", False)),
        augment=AC(enable=False),
        features=True,
        n_mels=int(feat_cfg["n_mels"]),
        n_fft=int(feat_cfg["n_fft"]),
        hop_length=int(feat_cfg["hop_length"]),
        win_length=int(feat_cfg["win_length"]),
        num_classes=num_classes,
    )

    train_loader = DataLoader(train_ds, batch_size=int(tr_cfg["batch_size"]), shuffle=True, num_workers=int(tr_cfg["num_workers"]), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=int(tr_cfg["batch_size"]), shuffle=False, num_workers=int(tr_cfg["num_workers"]), pin_memory=True)

    model = SimpleSpecCNN(n_mels=int(feat_cfg["n_mels"]), num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr_cfg["lr"]), weight_decay=float(tr_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tr_cfg["epochs"]))

    best_acc = -1.0
    history = []

    for epoch in range(1, int(tr_cfg["epochs"]) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{tr_cfg['epochs']}")
        total_loss = 0.0
        n = 0
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += float(loss.item()) * y.size(0)
            n += y.size(0)
            pbar.set_postfix(loss=total_loss/max(n, 1))

        scheduler.step()
        val_acc = evaluate_accuracy(model, val_loader, device)
        rec = {"epoch": epoch, "train_loss": total_loss/max(n,1), "val_acc": val_acc}
        history.append(rec)
        print(json.dumps(rec))

        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, os.path.join(out_dir, "last.pt"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, os.path.join(out_dir, "best.pt"))

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Done. best val_acc={best_acc:.4f}")
