from __future__ import annotations

import argparse

from watermark.data import list_audio_files, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", type=str, required=True, help="Root folder with audio files")
    ap.add_argument("--out", type=str, required=True, help="Output manifest CSV")
    ap.add_argument("--ext", type=str, nargs="+", default=["wav", "flac", "mp3", "ogg", "m4a", "aac"])
    args = ap.parse_args()

    paths = list_audio_files(args.audio_dir, exts=args.ext)
    if len(paths) == 0:
        raise SystemExit(f"No audio found under {args.audio_dir} with extensions {args.ext}")

    write_manifest(paths, args.out)
    print(f"Wrote {len(paths)} paths -> {args.out}")


if __name__ == "__main__":
    main()
