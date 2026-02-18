import os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


VOICE = "en-GB-RyanNeural"
BACKEND = "edge"
OUT_DIR = "my_wavs"
TEXTS_FILE = "texts.txt"
N = 1000
MAX_WORKERS = 12

def run_one(i, text):
    out_path = os.path.join(OUT_DIR, f"{i}.wav")
    cmd = [
        sys.executable, "app.py", "tts",
        "--backend", BACKEND,
        "--voice", VOICE,
        "--text", text,
        "--out", out_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return i, out_path, r.returncode, r.stderr

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(TEXTS_FILE, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()][:N]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(run_one, i, t) for i, t in enumerate(texts, start=1)]
        for fu in as_completed(futures):
            i, out_path, code, err = fu.result()
            if code != 0:
                print(f"[{i}] ERROR:\n{err}")
            else:
                print(f"[{i}] OK -> {out_path}")

if __name__ == "__main__":
    main()
