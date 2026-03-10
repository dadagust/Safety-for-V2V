from pathlib import Path
import csv

root = Path("data/LibriSpeech")

train_dirs = ["train-clean-100", "train-clean-360", "train-other-500"]
val_dirs   = ["dev-clean", "dev-other"]
test_dirs  = ["test-clean", "test-other"]

def list_flac(d: Path):
    return sorted(str(p) for p in d.rglob("*.flac"))

def collect(names):
    paths = []
    for n in names:
        d = root / n
        if d.exists():
            paths += list_flac(d)
    return paths

def write(paths, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path"])
        for p in paths:
            w.writerow([p])
    print(f"{out}: {len(paths)} files")

write(collect(train_dirs), "manifests/librispeech_train.csv")
write(collect(val_dirs),   "manifests/librispeech_val.csv")
write(collect(test_dirs),  "manifests/librispeech_test.csv")