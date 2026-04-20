# Audio Watermarking Thesis Kit v3

This project contains the waveform-domain speech watermarking prototype used in the thesis. The current codebase focuses on four practical improvements over the earlier starter version:

- a **message-conditioned dilated residual encoder** with a larger receptive field,
- a **localized detector** with a frame-level presence head and masked statistics pooling for payload decoding,
- a **stage-based curriculum** from bit-channel warm-up to localized, low-latency verification,
- reproducible scripts for **robustness sweeps**, **listening examples**, and **objective quality evaluation**.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional packages for objective perceptual evaluation:

```bash
pip install pesq pystoi
```

For MP3/AAC/Opus attacks you also need `ffmpeg` in `PATH`.

## Prepare manifests

```bash
python scripts/prepare_manifest.py --audio_dir data/LibriSpeech --out manifests/librispeech.csv --ext flac
python scripts/split_manifest.py --manifest manifests/librispeech.csv --out_dir manifests --val_ratio 0.02 --test_ratio 0.02
```

This creates:

- `manifests/librispeech_train.csv`
- `manifests/librispeech_val.csv`
- `manifests/librispeech_test.csv`

For faster experiments the repository also uses reduced manifests such as `train_2k.csv` and `val_200.csv`.

## Recommended training curriculum

### 0) Debug sanity check

Use this first to verify that the bit channel is learning at all.

```bash
python train.py --config configs/debug_stage0.yaml --train_manifest manifests/librispeech_train.csv --val_manifest manifests/librispeech_val.csv
```

### 1) Payload warm-up

Learn to decode message bits reliably before introducing negatives or localization.

```bash
python train.py --config configs/stage1_bridge_4bit.yaml --train_manifest manifests/librispeech_train.csv --val_manifest manifests/librispeech_val.csv
```

### 2) Detector calibration

Resume from stage 1 and teach the detector to separate watermarked and clean speech.

```bash
python train.py --config configs/stage2_detector_4bit.yaml --train_manifest manifests/librispeech_train.csv --val_manifest manifests/librispeech_val.csv --resume checkpoints_stage1/best_clean_bitacc.pt
```

### 3) Localization

Introduce watermark holes and train the localized presence head.

```bash
python train.py --config configs/stage3_localize_4bit.yaml --train_manifest manifests/librispeech_train.csv --val_manifest manifests/librispeech_val.csv --resume checkpoints_stage2/best_clean_bitacc.pt
```

### 4) Latency fine-tune

Fine-tune the localized checkpoint so that payload recovery happens earlier on 0.25/0.5/1.0/2.0 s prefixes.

```bash
python train.py --config configs/stage4_latency_4bit.yaml --train_manifest manifests/librispeech_train.csv --val_manifest manifests/librispeech_val.csv --resume checkpoints_stage3_localize_4bit/best_clean_bitacc.pt
```

### 5) Optional robustness variant

An additional robustness-oriented configuration is also included:

```bash
python train.py --config configs/variantA_stage4_robust.yaml --train_manifest manifests/librispeech_train.csv --val_manifest manifests/librispeech_val.csv --resume checkpoints_stage3_localize_4bit/best_clean_bitacc.pt
```

By default, `train.py --resume ...` loads **model weights only**. That is the recommended behavior for stage transitions. Use `--resume_optimizer` only when continuing the *same* stage.

## Evaluate

Clean evaluation:

```bash
python eval.py --config configs/stage4_latency_4bit.yaml --checkpoint checkpoints_stage4_latency_4bit/latest.pt --manifest manifests/librispeech_test.csv --no_attacks --embed_mode auto
```

Attacked evaluation:

```bash
python eval.py --config configs/stage4_latency_4bit.yaml --checkpoint checkpoints_stage4_latency_4bit/latest.pt --manifest manifests/librispeech_test.csv --embed_mode auto
```

## Robustness sweep and latency probes

```bash
python robustness_eval.py --config configs/stage4_latency_4bit.yaml --checkpoint checkpoints_stage4_latency_4bit/latest.pt --manifest manifests/librispeech_test.csv --out results/stage4_latency.csv --latency
```

The revised robustness path supports the original offline attack suite plus two additional channel families configured through `attacks_eval`:

- `opus_bitrates_k` for low-bitrate Opus round-trips,
- `phone_bandwidth_presets` for a PSTN-style narrowband proxy.

The default stage 3 / stage 4 configs now include:

```yaml
attacks_eval:
  opus_bitrates_k: [12, 24]
  phone_bandwidth_presets: [pstn]
```

## Objective quality evaluation

This renders watermarked audio from a checkpoint and computes clean-vs-watermarked metrics such as SNR, segmental SNR, mean/max absolute perturbation, and optionally PESQ/STOI if the extra packages are installed.

```bash
python quality_eval.py \
  --config configs/stage4_latency_4bit.yaml \
  --checkpoint checkpoints_stage4_latency_4bit/best_clean_bitacc.pt \
  --manifest manifests/librispeech_test.csv \
  --out_csv results/stage4_quality_rows.csv \
  --out_json results/stage4_quality_summary.json
```

Use `--max_items N` for a smaller smoke test.

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
python scripts/render_watermark_examples.py --config configs/stage4_latency_4bit.yaml --checkpoint checkpoints_stage4_latency_4bit/latest.pt --input data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac --out_dir examples/listen_001
```

Or from a manifest item:

```bash
python scripts/render_watermark_examples.py --config configs/stage4_latency_4bit.yaml --checkpoint checkpoints_stage4_latency_4bit/latest.pt --manifest manifests/librispeech_test.csv --index 0 --out_dir examples/listen_000
```

You can also fix the payload bits:

```bash
python scripts/render_watermark_examples.py --config configs/stage4_latency_4bit.yaml --checkpoint checkpoints_stage4_latency_4bit/latest.pt --manifest manifests/librispeech_test.csv --index 0 --bits 0101 --out_dir examples/listen_fixed_bits
```

If CPU preview is slow, render a shorter segment:

```bash
python scripts/render_watermark_examples.py --config configs/stage4_latency_4bit.yaml --checkpoint checkpoints_stage4_latency_4bit/latest.pt --manifest manifests/librispeech_test.csv --index 0 --segment_seconds_override 1.5 --out_dir examples/listen_short
```

## Main files

- `watermark/model.py` - encoder and detector
- `watermark/runtime.py` - shared helpers for model construction and decoding
- `watermark/attacks.py` - black-box evaluation attacks, including MP3/AAC/Opus and phone-band proxy
- `train.py` - staged training loop
- `eval.py` - clean/attacked evaluation
- `robustness_eval.py` - attack sweep and prefix probes
- `quality_eval.py` - paired quality metrics for clean vs watermarked audio
- `scripts/render_watermark_examples.py` - listening examples
- `configs/*.yaml` - staged curriculum configs
