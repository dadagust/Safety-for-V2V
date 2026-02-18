from __future__ import annotations
import os, csv, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)

    idx = {}
    for p in audio_dir.rglob("*"):
        if p.suffix.lower() in [".wav", ".flac"]:
            idx[p.stem] = str(p)

    rows = []
    with open(args.protocol, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            file_id = parts[1]          # <-- ключевое: 2-я колонка
            key = parts[-1].lower()     # bonafide/spoof

            if file_id not in idx:
                continue
            label = 1 if key == "bonafide" else 0
            rows.append((idx[file_id], label))

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()
