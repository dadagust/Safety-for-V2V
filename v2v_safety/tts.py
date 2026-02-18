from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class TTSConfig:
    model_name: str
    speaker: Optional[str] = None   # для pyttsx3 можно использовать как voice id/name (опционально)
    language: Optional[str] = None
    sr: int = 22050

def _synthesize_pyttsx3(text: str, voice: Optional[str] = None) -> tuple[np.ndarray, int]:
    import tempfile
    import os
    import soundfile as sf
    import pyttsx3

    engine = pyttsx3.init()

    # optional: set voice
    if voice:
        for v in engine.getProperty("voices"):
            if voice.lower() in (v.id or "").lower() or voice.lower() in (v.name or "").lower():
                engine.setProperty("voice", v.id)
                break

    # save to wav
    td = tempfile.mkdtemp()
    out_path = os.path.join(td, "tts.wav")
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    engine.stop()

    wav, sr = sf.read(out_path, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    # normalize if needed
    peak = float(np.max(np.abs(wav)) + 1e-9)
    if peak > 1.0:
        wav = wav / peak

    return wav, int(sr)

def synthesize(text: str, cfg: TTSConfig) -> tuple[np.ndarray, int]:
    """
    Backends:
      - model_name == "pyttsx3": offline Windows SAPI5 (no espeak required)
      - otherwise: Coqui TTS (may require phonemizer/espeak)
    """
    if cfg.model_name.lower() in ["pyttsx3", "sapi", "sapi5", "windows"]:
        return _synthesize_pyttsx3(text, voice=cfg.speaker)

    # Coqui path (если потом захочешь вернуть)
    try:
        from TTS.api import TTS
    except Exception as e:
        raise RuntimeError("Coqui TTS not installed. Run: pip install TTS") from e

    tts = TTS(cfg.model_name)
    out_sr = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", None) or cfg.sr
    wav = tts.tts(text=text, speaker=cfg.speaker, language=cfg.language)
    return np.asarray(wav, dtype=np.float32), int(out_sr)
