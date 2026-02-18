from __future__ import annotations
import torch
from typing import Dict, Any

from v2v_safety.audio import load_wav, pad_or_crop, simple_vad_trim, mel_spectrogram_db
from v2v_safety.detector.models.cnn import SimpleSpecCNN

@torch.no_grad()
def predict_file(cfg: Dict[str, Any], ckpt_path: str, wav_path: str) -> dict:
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    audio_cfg = cfg["audio"]
    feat_cfg = cfg["features"]

    clip = load_wav(wav_path, target_sr=int(audio_cfg["sr"]), mono=True)
    wav = clip.wav
    if bool(audio_cfg.get("vad", False)):
        wav = simple_vad_trim(wav, clip.sr)
    wav = pad_or_crop(wav, clip.sr, float(audio_cfg["clip_seconds"]), random_crop=False)

    x = torch.from_numpy(wav).unsqueeze(0)
    feat = mel_spectrogram_db(
        x, sr=clip.sr,
        n_mels=int(feat_cfg["n_mels"]),
        n_fft=int(feat_cfg["n_fft"]),
        hop_length=int(feat_cfg["hop_length"]),
        win_length=int(feat_cfg["win_length"]),
    ).unsqueeze(1)  # (1,1,n_mels,T)

    model = SimpleSpecCNN(n_mels=int(feat_cfg["n_mels"]), num_classes=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    logits = model(feat.to(device))
    prob = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    return {"path": wav_path, "prob_spoof": float(prob[0]), "prob_bonafide": float(prob[1]), "pred": "bonafide" if prob[1] >= 0.5 else "spoof"}
