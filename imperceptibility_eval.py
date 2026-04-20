from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from watermark.data import load_audio
from watermark.runtime import build_model_from_cfg, embed_batch


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_checkpoint_bundle(checkpoint_path: str) -> Dict[str, Any]:
    return torch.load(checkpoint_path, map_location='cpu')


def read_manifest_paths(manifest_csv: str) -> List[str]:
    with open(manifest_csv, 'r', newline='', encoding='utf-8') as f:
        return [row['path'] for row in csv.DictReader(f)]


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


def weighted_snr_db(clean: np.ndarray, test: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> float:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        w = w.reshape(-1)
    if np.max(w) <= 1e-8:
        return float('nan')
    c = clean.astype(np.float64)
    n = (test - clean).astype(np.float64)
    sig = float(np.sum(w * (c ** 2)))
    err = float(np.sum(w * (n ** 2)))
    denom = float(np.sum(w)) + eps
    return 10.0 * math.log10((sig / denom + eps) / (err / denom + eps))


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


def mean_abs_weighted(x: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if np.max(w) <= 1e-8:
        return float('nan')
    v = np.abs(np.asarray(x, dtype=np.float64).reshape(-1))
    return float(np.sum(v * w) / (np.sum(w) + eps))


def stft_mag_np(x: np.ndarray, n_fft: int, hop: int, win: int) -> np.ndarray:
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).view(1, -1)
    window = torch.hann_window(win, dtype=torch.float32)
    X = torch.stft(
        xt,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=True,
        return_complex=True,
    )
    return torch.abs(X).squeeze(0).cpu().numpy()


def log_spectral_distance_db(clean: np.ndarray, test: np.ndarray, n_fft: int = 1024, hop: int = 256, win: int = 1024, eps: float = 1e-7) -> float:
    C = stft_mag_np(clean, n_fft=n_fft, hop=hop, win=win)
    T = stft_mag_np(test, n_fft=n_fft, hop=hop, win=win)
    lc = np.log10(np.maximum(C, eps))
    lt = np.log10(np.maximum(T, eps))
    return float(np.mean(np.sqrt(np.mean((lc - lt) ** 2, axis=0))))


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


def deterministic_bits(path: str, nbits: int, seed: int) -> np.ndarray:
    h = hashlib.sha256(f'{seed}::{path}'.encode('utf-8')).digest()
    bits: List[float] = []
    idx = 0
    while len(bits) < nbits:
        byte = h[idx % len(h)]
        for shift in range(8):
            bits.append(float((byte >> shift) & 1))
            if len(bits) >= nbits:
                break
        idx += 1
    return np.asarray(bits[:nbits], dtype=np.float32)


@dataclass
class EvalRun:
    name: str
    checkpoint: str
    config_override: str = ''


def parse_run_spec(spec: str) -> EvalRun:
    if '=' in spec:
        name, ckpt = spec.split('=', 1)
        return EvalRun(name=name.strip(), checkpoint=ckpt.strip())
    p = Path(spec)
    return EvalRun(name=p.stem, checkpoint=str(p))


def load_cfg_for_run(run: EvalRun, global_config: str = '') -> Dict[str, Any]:
    if run.config_override:
        return load_yaml(run.config_override)
    if global_config:
        return load_yaml(global_config)
    bundle = load_checkpoint_bundle(run.checkpoint)
    cfg = bundle.get('cfg', None)
    if cfg is None:
        raise ValueError(
            f'Checkpoint {run.checkpoint} does not contain cfg. Pass --config explicitly.'
        )
    return cfg


def build_model_for_run(cfg: Dict[str, Any], checkpoint_path: str, device: torch.device):
    model = build_model_from_cfg(cfg, device=device)
    bundle = load_checkpoint_bundle(checkpoint_path)
    model.load_state_dict(bundle['model'], strict=True)
    model.eval()
    return model


def compute_summary(rows: List[Dict[str, Any]], keys: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {'num_items': len(rows)}
    for key in keys:
        vals = []
        for row in rows:
            value = row.get(key, '')
            if value == '' or value is None:
                continue
            fv = float(value)
            if math.isnan(fv):
                continue
            vals.append(fv)
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[key] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Imperceptibility benchmark for one or more checkpoints.')
    ap.add_argument('--checkpoint', action='append', required=True, help='Checkpoint path or name=checkpoint_path. Repeat to compare multiple checkpoints.')
    ap.add_argument('--config', type=str, default='', help='Optional global config override. Usually unnecessary because cfg is read from checkpoint.')
    ap.add_argument('--manifest', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--max_items', type=int, default=100, help='0 = all items.')
    ap.add_argument('--modes', type=str, default='auto,full', help='Comma-separated embed modes to evaluate: auto,full,window,dropout')
    ap.add_argument('--seed', type=int, default=1337, help='Seed for deterministic bit assignment.')
    ap.add_argument('--bits', type=str, default='', help='Optional fixed payload bits, e.g. 0101.')
    args = ap.parse_args()

    runs = [parse_run_spec(item) for item in args.checkpoint]
    paths = read_manifest_paths(args.manifest)
    if args.max_items > 0:
        paths = paths[:args.max_items]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pesq_fn = try_import_pesq()
    stoi_fn = try_import_stoi()
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    if not modes:
        raise ValueError('At least one mode must be provided in --modes')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(1)

    comparison_rows: List[Dict[str, Any]] = []
    summary_bundle: Dict[str, Any] = {
        'manifest': args.manifest,
        'max_items': args.max_items,
        'seed': args.seed,
        'pesq_available': pesq_fn is not None,
        'stoi_available': stoi_fn is not None,
        'runs': [],
    }

    for run in runs:
        cfg = load_cfg_for_run(run, global_config=args.config)
        sr = int(cfg['audio']['sample_rate'])
        seg_len = int(round(float(cfg['audio']['segment_seconds']) * sr))
        wm_len = int(round(float(cfg['audio']['watermark_seconds']) * sr))
        nbits = int(cfg['model']['nbits'])
        model = build_model_for_run(cfg, run.checkpoint, device)

        fixed_bits: Optional[torch.Tensor] = None
        if args.bits:
            bits_str = args.bits.strip()
            if len(bits_str) != nbits or any(ch not in '01' for ch in bits_str):
                raise ValueError(f'--bits must have exactly {nbits} characters of 0/1 for run {run.name}')
            fixed_bits = torch.tensor([[float(ch) for ch in bits_str]], dtype=torch.float32, device=device)

        run_record: Dict[str, Any] = {
            'name': run.name,
            'checkpoint': run.checkpoint,
            'sample_rate': sr,
            'segment_seconds': float(cfg['audio']['segment_seconds']),
            'watermark_seconds': float(cfg['audio']['watermark_seconds']),
            'results': {},
        }

        for raw_mode in modes:
            mode = resolve_embed_mode(cfg, raw_mode)
            rows: List[Dict[str, Any]] = []
            iterator = tqdm(paths, desc=f'{run.name}:{mode}')
            for path in iterator:
                x_np, _ = load_audio(path, target_sr=sr)
                if x_np.shape[0] < seg_len:
                    x_np = np.pad(x_np, (0, seg_len - x_np.shape[0]), mode='constant')
                else:
                    x_np = x_np[:seg_len]

                x = torch.from_numpy(x_np.astype(np.float32)).to(device).view(1, 1, -1)
                if fixed_bits is not None:
                    bits = fixed_bits
                else:
                    bits_np = deterministic_bits(path=path, nbits=nbits, seed=args.seed)
                    bits = torch.from_numpy(bits_np).to(device).view(1, -1)

                with torch.no_grad():
                    y, mask = embed_batch(
                        x=x,
                        bits=bits,
                        model=model,
                        cfg=cfg,
                        sample_rate=sr,
                        watermark_len=wm_len,
                        p_embed=1.0,
                        embed_mode_override=mode,
                    )

                clean = x_np.astype(np.float32)
                watermarked = y.detach().cpu().numpy().reshape(-1).astype(np.float32)
                mask_np = mask.detach().cpu().numpy().reshape(-1).astype(np.float32)
                delta = watermarked - clean

                row: Dict[str, Any] = {
                    'path': path,
                    'mode': mode,
                    'wm_fraction': float(np.mean(mask_np)),
                    'snr_db_full': snr_db(clean, watermarked),
                    'snr_db_wm_region': weighted_snr_db(clean, watermarked, mask_np),
                    'segmental_snr_db_full': segmental_snr_db(clean, watermarked, sr),
                    'mean_abs_delta_full': float(np.mean(np.abs(delta))),
                    'mean_abs_delta_wm_region': mean_abs_weighted(delta, mask_np),
                    'max_abs_delta': float(np.max(np.abs(delta))),
                    'l2_mse': float(np.mean(delta.astype(np.float64) ** 2)),
                    'log_spectral_distance_db': log_spectral_distance_db(clean, watermarked),
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

            per_item_csv = out_dir / f'{run.name}__{mode}.csv'
            fieldnames = list(rows[0].keys()) if rows else ['path', 'mode']
            with per_item_csv.open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in rows:
                    w.writerow(row)

            summary = compute_summary(
                rows,
                keys=[
                    'wm_fraction',
                    'snr_db_full',
                    'snr_db_wm_region',
                    'segmental_snr_db_full',
                    'mean_abs_delta_full',
                    'mean_abs_delta_wm_region',
                    'max_abs_delta',
                    'l2_mse',
                    'log_spectral_distance_db',
                    'pesq_wb',
                    'stoi',
                ],
            )
            summary['mode'] = mode
            summary['per_item_csv'] = str(per_item_csv)
            run_record['results'][mode] = summary

            comp_row = {
                'name': run.name,
                'checkpoint': run.checkpoint,
                'mode': mode,
            }
            for key, value in summary.items():
                if isinstance(value, dict) and 'mean' in value:
                    comp_row[f'{key}_mean'] = value['mean']
                    comp_row[f'{key}_std'] = value['std']
            comparison_rows.append(comp_row)

        summary_bundle['runs'].append(run_record)

    summary_json = out_dir / 'imperceptibility_summary.json'
    summary_json.write_text(json.dumps(summary_bundle, indent=2), encoding='utf-8')

    comparison_csv = out_dir / 'imperceptibility_comparison.csv'
    if comparison_rows:
        fieldnames: List[str] = []
        for row in comparison_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with comparison_csv.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in comparison_rows:
                w.writerow(row)

    print('Wrote:', summary_json)
    print('Wrote:', comparison_csv)
    print(json.dumps(summary_bundle, indent=2))
    if pesq_fn is None:
        print('Optional metric unavailable: PESQ (install `pesq`)')
    if stoi_fn is None:
        print('Optional metric unavailable: STOI (install `pystoi`)')


if __name__ == '__main__':
    main()
