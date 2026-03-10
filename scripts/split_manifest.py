from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List

from watermark.data import read_manifest, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--test_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    paths = read_manifest(args.manifest)
    rng = random.Random(args.seed)
    rng.shuffle(paths)

    n = len(paths)
    n_test = int(round(n * args.test_ratio))
    n_val = int(round(n * args.val_ratio))
    n_train = n - n_val - n_test
    train = paths[:n_train]
    val = paths[n_train : n_train + n_val]
    test = paths[n_train + n_val :]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.manifest).stem
    write_manifest(train, out_dir / f"{stem}_train.csv")
    write_manifest(val, out_dir / f"{stem}_val.csv")
    write_manifest(test, out_dir / f"{stem}_test.csv")

    print(f"Total={n} train={len(train)} val={len(val)} test={len(test)} -> {out_dir}")


if __name__ == "__main__":
    main()
