# Audio Watermarking (thesis starter kit)

This is a **research / thesis** starter project for learning an **inaudible, robust audio watermark** with:
- an **encoder (watermark generator)** that embeds a secret bitstring into audio,
- a **detector/decoder** that can localize watermark presence and recover the bitstring,
- training-time **differentiable attacks** (noise, filtering, resampling, cropping) and
- evaluation-time **non‑differentiable attacks** (e.g., MP3/AAC via ffmpeg).

The design is inspired by public descriptions of Google DeepMind **SynthID** (inaudible watermark robust to common edits) and by open research like **AudioSeal** (localized watermark detection).  
See the accompanying notes in the assistant message for dataset suggestions.

## Quick start

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: this project intentionally avoids `torchaudio` / `librosa` to reduce system dependency issues.

### 2) Prepare a folder of audio files
Put `.wav`, `.flac`, `.mp3`, `.ogg` etc under:
```
data/audio/...
```

Then build manifests:
```bash
python scripts/prepare_manifest.py --audio_dir data/audio --out manifests/train.csv --ext wav flac mp3 ogg
python scripts/split_manifest.py --manifest manifests/train.csv --out_dir manifests --val_ratio 0.02 --test_ratio 0.02
```

### 3) Train
```bash
python train.py --config configs/base.yaml --train_manifest manifests/train_train.csv --val_manifest manifests/train_val.csv
```

#### Variant A (localized + low-latency) training (recommended)

Stage 1 (no time-warp yet):
```bash
python train.py --config configs/variantA_stage1.yaml --train_manifest manifests/train_train.csv --val_manifest manifests/train_val.csv
```

Stage 2 (resume + add speed/crop attacks):
```bash
python train.py --config configs/variantA_stage2.yaml --train_manifest manifests/train_train.csv --val_manifest manifests/train_val.csv --resume checkpoints/latest.pt
```

### 4) Evaluate
```bash
python eval.py --config configs/base.yaml --checkpoint checkpoints/latest.pt --manifest manifests/train_test.csv
```

For localized embedding + localization metrics:
```bash
python eval.py --config configs/variantA_stage2.yaml --checkpoint checkpoints/latest.pt --manifest manifests/train_test.csv --embed_mode dropout
```

### 5) Robustness sweep (attacks)
```bash
python robustness_eval.py --config configs/base.yaml --checkpoint checkpoints/latest.pt --manifest manifests/train_test.csv --out results/robustness.csv
```

> Use the **same config** you trained with (it defines model size and number of bits).

## Project layout
- `watermark/model.py` – encoder + detector networks
- `watermark/attacks.py` – differentiable training attacks + evaluation attacks (incl. ffmpeg codecs)
- `watermark/losses.py` – multi‑scale STFT loss + helpers
- `watermark/data.py` – manifest dataset + audio I/O/resampling
- `train.py` – training loop
- `eval.py` – basic evaluation (TPR/FPR + bit accuracy)
- `robustness_eval.py` – attack sweep + latency probing
- `configs/base.yaml` – hyperparameters

## Notes
- To claim “better than SynthID”, you must define the benchmark: e.g., **detection latency**, **bit accuracy after attacks**, and **perceptual quality**.
- MP3/AAC attacks use `ffmpeg` and temporary files; you can disable codecs in config if needed.
