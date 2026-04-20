from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from watermark.data import load_audio
from watermark.runtime import build_model_from_cfg, embed_batch


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(cfg: Dict[str, Any], checkpoint_path: str, device: torch.device):
    model = build_model_from_cfg(cfg, device=device)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    return model


def read_manifest_paths(manifest_csv: str) -> List[str]:
    with open(manifest_csv, 'r', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        return [row['path'] for row in r]


def resolve_embed_mode(cfg: Dict[str, Any], arg_mode: str) -> str:
    if arg_mode != 'auto':
        return arg_mode
    mode = str((cfg.get('embed', {}) or {}).get('mode', 'window')).lower()
    if mode in ('full', 'window', 'dropout', 'dropout_holes', 'holes'):
        return 'dropout' if mode in ('dropout_holes', 'holes') else mode
    return 'window'


def snr_db(clean: np.ndarray, test: np.ndarray, eps: float = 1e-12) -> float:
    noise = test - clean
    sig = float(np.sum(clean.astype(np.float64) ** 2))
    err = float(np.sum(noise.astype(np.float64) ** 2))
    return 10.0 * math.log10((sig + eps) / (err + eps))


def segmental_snr_db(clean: np.ndarray, test: np.ndarray, sr: int, frame_ms: float = 30.0, eps: float = 1e-12) -> float:
    frame_len = max(1, int(round(frame_ms * sr / 1000.0)))
    vals: List[float] = []
    for start in range(0, clean.shape[0], frame_len):
        c = clean[start:start + frame_len]
        t = test[start:start + frame_len]
        if c.size == 0:
            continue
        sig = float(np.sum(c.astype(np.float64) ** 2))
        err = float(np.sum((t - c).astype(np.float64) ** 2))
        vals.append(10.0 * math.log10((sig + eps) / (err + eps)))
    return float(np.mean(vals)) if vals else float('nan')


def try_import_pesq():
    try:
        from pesq import pesq  # type: ignore
        return pesq
    except Exception:
        return None


def try_import_stoi():
    try:
        from pystoi import stoi  # type: ignore
        return stoi
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description='Objective clean-vs-watermarked quality evaluation.')
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--manifest', type=str, required=True)
    ap.add_argument('--out_csv', type=str, required=True)
    ap.add_argument('--out_json', type=str, default='')
    ap.add_argument('--max_items', type=int, default=100, help='Limit items for speed. 0 = all.')
    ap.add_argument('--embed_mode', type=str, default='auto', choices=['auto', 'full', 'window', 'dropout'])
    ap.add_argument('--bits', type=str, default='', help='Optional fixed payload bits, e.g. 0101. Defaults to random.')
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(1)
    sr = int(cfg['audio']['sample_rate'])
    seg_len = int(round(float(cfg['audio']['segment_seconds']) * sr))
    wm_len = int(round(float(cfg['audio']['watermark_seconds']) * sr))
    nbits = int(cfg['model']['nbits'])
    embed_mode = resolve_embed_mode(cfg, args.embed_mode)

    model = load_model(cfg, args.checkpoint, device)
    paths = read_manifest_paths(args.manifest)
    if args.max_items and args.max_items > 0:
        paths = paths[: args.max_items]

    fixed_bits: Optional[torch.Tensor] = None
    if args.bits:
        bits_str = args.bits.strip()
        if len(bits_str) != nbits or any(ch not in '01' for ch in bits_str):
            raise ValueError(f'--bits must have exactly {nbits} characters of 0/1')
        fixed_bits = torch.tensor([[float(ch) for ch in bits_str]], dtype=torch.float32, device=device)

    pesq_fn = try_import_pesq()
    stoi_fn = try_import_stoi()

    rows: List[Dict[str, Any]] = []
    for path in tqdm(paths, desc='quality'):
        x_np, _ = load_audio(path, target_sr=sr)
        if x_np.shape[0] < seg_len:
            x_np = np.pad(x_np, (0, seg_len - x_np.shape[0]), mode='constant')
        else:
            x_np = x_np[:seg_len]

        x = torch.from_numpy(x_np).to(device).view(1, 1, -1)
        bits = fixed_bits if fixed_bits is not None else torch.randint(0, 2, (1, nbits), device=device, dtype=torch.float32)

        with torch.no_grad():
            y, _mask = embed_batch(
                x=x,
                bits=bits,
                model=model,
                cfg=cfg,
                sample_rate=sr,
                watermark_len=wm_len,
                p_embed=1.0,
                embed_mode_override=embed_mode,
            )

        clean = x_np.astype(np.float32)
        watermarked = y.detach().cpu().numpy().reshape(-1).astype(np.float32)
        delta = watermarked - clean

        row: Dict[str, Any] = {
            'path': path,
            'snr_db': snr_db(clean, watermarked),
            'segmental_snr_db': segmental_snr_db(clean, watermarked, sr),
            'mean_abs_delta': float(np.mean(np.abs(delta))),
            'max_abs_delta': float(np.max(np.abs(delta))),
            'l2_mse': float(np.mean(delta.astype(np.float64) ** 2)),
        }

        if pesq_fn is not None:
            try:
                row['pesq_wb'] = float(pesq_fn(sr, clean, watermarked, 'wb'))
            except Exception:
                row['pesq_wb'] = ''
        else:
            row['pesq_wb'] = ''

        if stoi_fn is not None:
            try:
                row['stoi'] = float(stoi_fn(clean, watermarked, sr, extended=False))
            except Exception:
                row['stoi'] = ''
        else:
            row['stoi'] = ''

        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ['path']
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    numeric_keys = ['snr_db', 'segmental_snr_db', 'mean_abs_delta', 'max_abs_delta', 'l2_mse', 'pesq_wb', 'stoi']
    missing_optional_metrics: List[str] = []
    if pesq_fn is None:
        missing_optional_metrics.append('pesq_wb')
    if stoi_fn is None:
        missing_optional_metrics.append('stoi')

    summary: Dict[str, Any] = {
        'num_items': len(rows),
        'sample_rate': sr,
        'embed_mode': embed_mode,
        'pesq_available': pesq_fn is not None,
        'stoi_available': stoi_fn is not None,
        'missing_optional_metrics': missing_optional_metrics,
    }
    for key in numeric_keys:
        vals = []
        for row in rows:
            value = row.get(key, '')
            if value == '' or value is None:
                continue
            vals.append(float(value))
        if vals:
            summary[key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
            }

    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix('.summary.json')
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('Wrote:', out_csv)
    print('Wrote:', out_json)
    print(json.dumps(summary, indent=2))
    if missing_optional_metrics:
        print('Missing optional metrics:', ', '.join(missing_optional_metrics))


if __name__ == '__main__':
    main()
