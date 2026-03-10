# Audio Watermarking Thesis Kit v3

This version upgrades the old starter kit in four important ways:

- **Better encoder**: a message-conditioned **dilated residual encoder** with a much larger receptive field.
- **Better detector**: a **localized presence head** plus a **global masked bit decoder** (mean+std pooling over the watermark region), which is far easier to train than per-frame bit logits.
- **Stage-based configs**: a practical curriculum from easy payload learning to localized, low-latency, attack-robust watermarking.
- **Listening examples**: a script that writes `original.wav`, `watermarked.wav`, and amplified `delta` files so you can hear how the watermark changes the signal.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For MP3/AAC robustness attacks you also need `ffmpeg` in `PATH`.

## Prepare manifests

```bash
python scripts/prepare_manifest.py --audio_dir data/LibriSpeech --out manifests/librispeech.csv --ext flac
python scripts/split_manifest.py --manifest manifests/librispeech.csv --out_dir manifests --val_ratio 0.02 --test_ratio 0.02
```

This creates:
- `manifests/librispeech_train.csv`
- `manifests/librispeech_val.csv`
- `manifests/librispeech_test.csv`

## Recommended training curriculum

### 0) Debug sanity check
Use this first if you want to verify that the bit channel is learning at all.

```bash
python train.py --config configs/debug_stage0.yaml --train_manifest manifests/librispeech_train.csv --val_manifest manifests/librispeech_val.csv
```

### 1) Payload warm-up
Learn to decode message bits reliably before strong localization/latency constraints.

```bash
python train.py --config configs/stage1_bridge_4bit.yaml --train_manifest manifests/librispeech_train.csv --val_manifest manifests/librispeech_val.csv
```

### 2) Localized watermark warm-up
Resume from stage 1, but now the watermark becomes localized by adding holes.

```bash
python train.py --config configs/stage2_detector_4bit.yaml --train_manifest manifests/librispeech_train.csv --val_manifest manifests/librispeech_val.csv --resume checkpoints_stage1/best_clean_bitacc.pt
```

### 3) Localized + lower-latency + mild time-warp

```bash
python train.py --config configs/stage3_localize_4bit.yaml --train_manifest manifests/librispeech_train.csv --val_manifest manifests/librispeech_val.csv --resume checkpoints_stage2/best_clean_bitacc.pt
```

### 4) Optional stronger robustness fine-tune

```bash
python train.py --config configs/variantA_stage4_robust.yaml --train_manifest manifests/librispeech_train.csv --val_manifest manifests/librispeech_val.csv --resume checkpoints_stage3/best_clean_bitacc.pt
```

By default, `train.py --resume ...` loads **model weights only**. That is the recommended behavior for stage transitions. Use `--resume_optimizer` only when continuing the *same* stage.

## Evaluate

Clean evaluation:

```bash
python eval.py --config configs/stage3_localize_4bit.yaml --checkpoint checkpoints_stage3/latest.pt --manifest manifests/librispeech_test.csv --no_attacks --embed_mode auto
```

Attacked evaluation:

```bash
python eval.py --config configs/stage3_localize_4bit.yaml --checkpoint checkpoints_stage3/latest.pt --manifest manifests/librispeech_test.csv --embed_mode auto
```

## Robustness sweep

```bash
python robustness_eval.py --config configs/stage3_localize_4bit.yaml --checkpoint checkpoints_stage3/latest.pt --manifest manifests/librispeech_test.csv --out results/robustness.csv --latency
```

## Listening examples

This writes:
- `original.wav`
- `watermarked.wav`
- `delta.wav`
- `delta_normalized.wav`
- `delta_xgain.wav`
- `meta.json`

Example with a direct input file:

```bash
python scripts/render_watermark_examples.py --config configs/stage3_localize_4bit.yaml --checkpoint checkpoints_stage3/latest.pt --input data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac --out_dir examples/listen_001
```

Or from a manifest item:

```bash
python scripts/render_watermark_examples.py --config configs/stage3_localize_4bit.yaml --checkpoint checkpoints_stage3/latest.pt --manifest manifests/librispeech_test.csv --index 0 --out_dir examples/listen_000
```

You can also fix the payload bits:

```bash
python scripts/render_watermark_examples.py --config configs/stage3_localize_4bit.yaml --checkpoint checkpoints_stage3/latest.pt --manifest manifests/librispeech_test.csv --index 0 --bits 1011001110001111 --out_dir examples/listen_fixed_bits
```

If CPU preview is slow, render a shorter segment:

```bash
python scripts/render_watermark_examples.py --config configs/stage3_localize_4bit.yaml --checkpoint checkpoints_stage3/latest.pt --manifest manifests/librispeech_test.csv --index 0 --segment_seconds_override 1.5 --out_dir examples/listen_short
```

## Main files

- `watermark/model.py` – upgraded encoder and detector
- `watermark/runtime.py` – shared helpers for model construction and decoding
- `train.py` – upgraded training loop with clean + attacked validation
- `eval.py` – evaluation with localization metrics
- `robustness_eval.py` – attack sweep and latency probes
- `scripts/render_watermark_examples.py` – listening examples
- `configs/*.yaml` – staged curriculum configs
