from __future__ import annotations
import os, csv, argparse, random
from pathlib import Path

from v2v_safety.tts import TTSConfig, synthesize
from v2v_safety.watermark import WatermarkConfig, embed_zero_bit_spread_spectrum
from v2v_safety.audio import save_wav, load_wav

DEFAULT_TEXTS = [
    "Hello! This is a test sentence for the assistant.",
    "Please confirm your identity before proceeding.",
    "The weather tomorrow will be sunny with light wind.",
    "I can help you book a ticket and set reminders.",
    "This audio is generated for research on watermarking.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="toy_data")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tts-cfg", default="configs/tts.yaml")
    ap.add_argument("--wm-cfg", default="configs/watermark.yaml")
    args = ap.parse_args()

    import yaml
    tcfg = TTSConfig(**yaml.safe_load(open(args.tts_cfg, "r", encoding="utf-8")))
    wcfg = WatermarkConfig(**yaml.safe_load(open(args.wm_cfg, "r", encoding="utf-8")))

    out_dir = Path(args.out_dir)
    (out_dir/"wav").mkdir(parents=True, exist_ok=True)
    train_csv = out_dir/"train.csv"
    val_csv = out_dir/"val.csv"

    rows = []
    for i in range(args.n):
        text = random.choice(DEFAULT_TEXTS)
        wav, sr = synthesize(text, tcfg)
        clean_path = out_dir/"wav"/f"bonafide_{i:04d}.wav"
        save_wav(str(clean_path), wav, sr)

        clip = load_wav(str(clean_path), target_sr=wcfg.sr)
        wm = embed_zero_bit_spread_spectrum(clip.wav, wcfg)
        spoof_path = out_dir/"wav"/f"spoof_{i:04d}.wav"
        save_wav(str(spoof_path), wm, wcfg.sr)

        rows.append((str(clean_path), 1))
        rows.append((str(spoof_path), 0))

    random.shuffle(rows)
    split = int(0.8 * len(rows))
    train_rows = rows[:split]
    val_rows = rows[split:]

    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(train_rows)

    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(val_rows)

    print("Toy dataset created:")
    print("  ", train_csv)
    print("  ", val_csv)
    print("NOTE: This is ONLY to smoke-test the pipeline quickly (watermarked TTS treated as spoof).")

if __name__ == "__main__":
    main()
