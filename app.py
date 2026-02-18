from __future__ import annotations

import argparse
import os
import json
import tempfile
from typing import Any, Dict, Tuple, Optional

import yaml
import numpy as np

from v2v_safety.audio import load_wav, save_wav
from v2v_safety.watermark import (
    WatermarkConfig,
    embed_zero_bit_spread_spectrum,
    detect_zero_bit_spread_spectrum,
)

# Оставляем совместимость со старым бэкендом (если он у тебя есть)
from v2v_safety.tts import TTSConfig, synthesize

from v2v_safety.detector.train import train_detector
from v2v_safety.detector.eval import eval_eer
from v2v_safety.detector.infer import predict_file


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _write_tmp_wav(wav: np.ndarray, sr: int) -> str:
    """
    Пишем временный wav на диск и возвращаем путь.
    Потом обязательно удалить файл.
    """
    import soundfile as sf

    fd, p = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:  # стерео -> моно
        wav = wav.mean(axis=1)

    sf.write(p, wav, sr)
    return p


def _synthesize_edge_tts(
    text: str,
    voice: str = "ru-RU-SvetlanaNeural",
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    output_format: str = "riff-24khz-16bit-mono-pcm",
) -> Tuple[np.ndarray, int]:
    """
    Реалистичный TTS на Windows через Microsoft Edge TTS (online).
    Требует: pip install edge-tts soundfile
    """
    try:
        import asyncio
        import edge_tts
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(
            "Не найден пакет для Edge TTS. Установи:\n"
            "  pip install edge-tts soundfile\n"
        ) from e

    async def _run(out_path: str):
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
        )
        await communicate.save(out_path)

    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        asyncio.run(_run(out_path))
        wav, sr = sf.read(out_path, dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        return np.asarray(wav, dtype=np.float32), int(sr)
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def _list_edge_voices() -> list[dict]:
    try:
        import asyncio
        import edge_tts
    except ImportError as e:
        raise RuntimeError(
            "Не найден пакет edge-tts. Установи:\n"
            "  pip install edge-tts\n"
        ) from e

    async def _run():
        return await edge_tts.list_voices()

    return asyncio.run(_run())


def _synthesize_coqui_tts(
    text: str,
    model_name: str,
    speaker: Optional[str] = None,
    language: Optional[str] = None,
    gpu: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Офлайн TTS через Coqui (качество зависит от модели).
    Требует: pip install coqui-tts (+ torch отдельно)
    """
    try:
        from TTS.api import TTS as CoquiTTS
    except ImportError as e:
        raise RuntimeError(
            "Не найден пакет Coqui TTS. Установи:\n"
            "  pip install coqui-tts\n"
            "И torch (CPU пример):\n"
            "  pip install --index-url https://download.pytorch.org/whl/cpu torch\n"
        ) from e

    tts = CoquiTTS(model_name=model_name, progress_bar=False, gpu=gpu)
    wav = tts.tts(text, speaker=speaker, language=language)
    # у разных версий API может отличаться, но обычно так:
    sr = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", None)
    if sr is None:
        # fallback: часто 22050
        sr = 22050
    return np.asarray(wav, dtype=np.float32), int(sr)


def cmd_tts(args):
    tcfg = load_yaml(args.tts_cfg)
    wcfg = load_yaml(args.watermark_cfg)

    # backend: CLI > yaml > auto/native
    backend = (args.backend or "").lower()
    if backend in ("", "auto"):
        backend = str(tcfg.get("backend", "native")).lower()

    if args.list_voices:
        voices = _list_edge_voices()
        if args.voice_filter:
            vf = args.voice_filter.lower()
            voices = [v for v in voices if vf in (v.get("ShortName", "").lower() + " " + v.get("Locale", "").lower())]
        print(json.dumps(voices, ensure_ascii=False, indent=2))
        return

    if backend == "edge":
        voice = args.voice or tcfg.get("voice", "ru-RU-SvetlanaNeural")
        rate = args.rate or tcfg.get("rate", "+0%")
        volume = args.volume or tcfg.get("volume", "+0%")
        pitch = args.pitch or tcfg.get("pitch", "+0Hz")
        output_format = tcfg.get("output_format", "riff-24khz-16bit-mono-pcm")

        wav, sr = _synthesize_edge_tts(
            args.text,
            voice=str(voice),
            rate=str(rate),
            volume=str(volume),
            pitch=str(pitch),
            output_format=str(output_format),
        )

    elif backend == "coqui":
        model_name = args.model_name or tcfg.get("model_name")
        if not model_name:
            raise RuntimeError(
                "Для backend=coqui нужен model_name (в configs/tts.yaml или через --model-name)."
            )
        wav, sr = _synthesize_coqui_tts(
            args.text,
            model_name=str(model_name),
            speaker=args.speaker or tcfg.get("speaker"),
            language=args.language or tcfg.get("language"),
            gpu=bool(tcfg.get("gpu", False)),
        )

    else:
        wav, sr = synthesize(args.text, TTSConfig(**tcfg))

    if args.watermark:
        cfg = WatermarkConfig(**wcfg)
        tmp_path = _write_tmp_wav(wav, sr)
        try:
            clip = load_wav(tmp_path, target_sr=int(cfg.sr))
            out = embed_zero_bit_spread_spectrum(clip.wav, cfg)
            save_wav(args.out, out, cfg.sr)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    else:
        save_wav(args.out, wav, sr)

    print(f"Saved: {args.out}")


def cmd_wm_embed(args):
    wcfg = WatermarkConfig(**load_yaml(args.watermark_cfg))
    clip = load_wav(args.input, target_sr=wcfg.sr)
    out = embed_zero_bit_spread_spectrum(clip.wav, wcfg)
    save_wav(args.out, out, wcfg.sr)
    print(f"Saved: {args.out}")


def cmd_wm_detect(args):
    wcfg = WatermarkConfig(**load_yaml(args.watermark_cfg))
    clip = load_wav(args.input, target_sr=wcfg.sr)
    present, score = detect_zero_bit_spread_spectrum(clip.wav, wcfg)
    print(
        json.dumps(
            {"present": bool(present), "score": float(score), "threshold": float(wcfg.threshold)},
            indent=2,
        )
    )


def cmd_det_train(args):
    cfg = load_yaml(args.detector_cfg)
    train_detector(cfg, args.train_csv, args.val_csv, args.out_dir, num_classes=2)


def cmd_det_eval(args):
    cfg = load_yaml(args.detector_cfg)
    eval_eer(cfg, args.csv, args.ckpt, args.out_dir)


def cmd_det_infer(args):
    cfg = load_yaml(args.detector_cfg)
    res = predict_file(cfg, args.ckpt, args.wav)
    print(json.dumps(res, indent=2))


def main():
    p = argparse.ArgumentParser("v2v_safety app")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("tts", help="Generate TTS (optionally watermarked)")
    s.add_argument("--text", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tts-cfg", default="configs/tts.yaml")
    s.add_argument("--watermark-cfg", default="configs/watermark.yaml")
    s.add_argument("--watermark", action="store_true")

    # NEW: выбор бэкенда TTS
    s.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "native", "edge", "coqui"],
        help="TTS backend: auto(yaml), native(old), edge(realistic online), coqui(offline)",
    )

    # NEW: Edge TTS overrides
    s.add_argument("--voice", default=None, help="Edge voice, e.g. ru-RU-SvetlanaNeural")
    s.add_argument("--rate", default=None, help="Edge rate, e.g. +0%, -10%")
    s.add_argument("--volume", default=None, help="Edge volume, e.g. +0%, -10%")
    s.add_argument("--pitch", default=None, help="Edge pitch, e.g. +0Hz, +20Hz")
    s.add_argument("--list-voices", action="store_true", help="List Edge TTS voices and exit")
    s.add_argument("--voice-filter", default=None, help="Filter for --list-voices (substring)")

    # NEW: Coqui overrides
    s.add_argument("--model-name", default=None, help="Coqui model name (required for backend=coqui)")
    s.add_argument("--speaker", default=None, help="Coqui speaker (optional)")
    s.add_argument("--language", default=None, help="Coqui language (optional)")

    s.set_defaults(fn=cmd_tts)

    s = sub.add_parser("wm-embed", help="Embed watermark into an existing wav")
    s.add_argument("--input", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--watermark-cfg", default="configs/watermark.yaml")
    s.set_defaults(fn=cmd_wm_embed)

    s = sub.add_parser("wm-detect", help="Detect watermark in wav")
    s.add_argument("--input", required=True)
    s.add_argument("--watermark-cfg", default="configs/watermark.yaml")
    s.set_defaults(fn=cmd_wm_detect)

    s = sub.add_parser("det-train", help="Train detector on CSV")
    s.add_argument("--train-csv", required=True)
    s.add_argument("--val-csv", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--detector-cfg", default="configs/detector.yaml")
    s.set_defaults(fn=cmd_det_train)

    s = sub.add_parser("det-eval", help="Evaluate EER + DET curve on CSV")
    s.add_argument("--csv", required=True)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--detector-cfg", default="configs/detector.yaml")
    s.set_defaults(fn=cmd_det_eval)

    s = sub.add_parser("det-infer", help="Infer on one wav")
    s.add_argument("--wav", required=True)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--detector-cfg", default="configs/detector.yaml")
    s.set_defaults(fn=cmd_det_infer)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
