from __future__ import annotations
import os, json
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from v2v_safety.detector.models.cnn import SimpleSpecCNN
from v2v_safety.detector.data import AudioCSVDataset, AugmentConfig
from v2v_safety.detector.eer import compute_eer

@torch.no_grad()
def eval_eer(cfg: Dict[str, Any], csv_path: str, ckpt_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    audio_cfg = cfg["audio"]
    feat_cfg = cfg["features"]
    tr_cfg = cfg["train"]

    ds = AudioCSVDataset(
        csv_path,
        sr=int(audio_cfg["sr"]),
        clip_seconds=float(audio_cfg["clip_seconds"]),
        vad=bool(audio_cfg.get("vad", False)),
        augment=AugmentConfig(enable=False),
        features=True,
        n_mels=int(feat_cfg["n_mels"]),
        n_fft=int(feat_cfg["n_fft"]),
        hop_length=int(feat_cfg["hop_length"]),
        win_length=int(feat_cfg["win_length"]),
        num_classes=2,
    )
    loader = DataLoader(ds, batch_size=int(tr_cfg["batch_size"]), shuffle=False, num_workers=int(tr_cfg["num_workers"]), pin_memory=True)

    model = SimpleSpecCNN(n_mels=int(feat_cfg["n_mels"]), num_classes=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    y_true, y_score = [], []
    for x, y in tqdm(loader, desc="eval"):
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1]  # class 1 = bonafide
        y_true.append(y.numpy())
        y_score.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(y_true).astype(np.int32)
    y_score = np.concatenate(y_score).astype(np.float32)
    eer = compute_eer(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr

    plt.figure()
    plt.plot(fpr, fnr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title(f"DET (approx). EER={eer*100:.2f}%")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "det_curve.png"), dpi=160, bbox_inches="tight")

    with open(os.path.join(out_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump({"eer": eer, "eer_percent": eer*100.0}, f, indent=2)
    print(f"EER={eer*100:.2f}% -> {out_dir}")
