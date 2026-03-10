from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .embed import embed_dropout_holes, embed_full, embed_random_window
from .model import Detector1D, Encoder1D, WatermarkNet


def filter_kwargs_for_ctor(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys())
    valid.discard("self")
    return {k: v for k, v in kwargs.items() if k in valid}


def build_model_from_cfg(cfg: Dict[str, Any], device: torch.device | None = None) -> WatermarkNet:
    enc_kwargs = filter_kwargs_for_ctor(
        Encoder1D,
        {
            "nbits": int(cfg["model"]["nbits"]),
            "msg_dim": int(cfg["model"]["msg_dim"]),
            "hidden": int(cfg["model"]["enc_hidden"]),
            "max_delta": float(cfg["model"]["max_delta"]),
            "kernel_size": int(cfg["model"].get("enc_kernel_size", 7)),
            "dilations": list(cfg["model"].get("enc_dilations", [1, 2, 4, 8, 16, 32, 64])),
            "groups": int(cfg["model"].get("enc_groups", 8)),
            "dropout": float(cfg["model"].get("enc_dropout", 0.0)),
        },
    )
    det_kwargs = filter_kwargs_for_ctor(
        Detector1D,
        {
            "nbits": int(cfg["model"]["nbits"]),
            "channels": list(cfg["model"]["det_channels"]),
            "kernel_size": int(cfg["model"]["det_kernel_size"]),
            "strides": list(cfg["model"]["det_strides"]),
            "bit_mlp_hidden": int(cfg["model"].get("bit_mlp_hidden", 256)),
            "groups": int(cfg["model"].get("det_groups", 8)),
            "dropout": float(cfg["model"].get("det_dropout", 0.0)),
        },
    )
    model = WatermarkNet(Encoder1D(**enc_kwargs), Detector1D(**det_kwargs))
    if device is not None:
        model = model.to(device)
    return model


@torch.no_grad()
def downsample_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    m = F.interpolate(mask, size=target_len, mode="linear", align_corners=False)
    return m.squeeze(1).clamp(0.0, 1.0)


@torch.no_grad()
def presence_to_weights(
    presence_logits: torch.Tensor,
    threshold: float = 0.5,
    use_soft_mask: bool = True,
    topk_fraction: float = 0.1,
) -> torch.Tensor:
    prob = torch.sigmoid(presence_logits)
    if use_soft_mask:
        return prob.clamp(0.0, 1.0)

    w = (prob >= float(threshold)).float()
    empty = w.sum(dim=-1) < 1.0
    if bool(empty.any().item()):
        B, T = w.shape
        k = max(1, int(round(float(topk_fraction) * T)))
        topk = torch.topk(prob, k=k, dim=-1).indices
        for b in range(B):
            if bool(empty[b].item()):
                w[b].zero_()
                w[b, topk[b]] = 1.0
    return w


def detector_forward_with_gt_mask(
    detector: Detector1D,
    audio_t: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    presence_logits, features = detector.forward_features(audio_t)
    mask_ds = downsample_mask(mask, features.shape[-1])
    bit_logits = detector.decode_bits(features, mask_ds)
    return presence_logits, bit_logits, mask_ds


@torch.no_grad()
def detector_forward_with_predicted_mask(
    detector: Detector1D,
    audio_t: torch.Tensor,
    threshold: float = 0.5,
    use_soft_mask: bool = True,
    topk_fraction: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    presence_logits, features = detector.forward_features(audio_t)
    weights = presence_to_weights(
        presence_logits,
        threshold=threshold,
        use_soft_mask=use_soft_mask,
        topk_fraction=topk_fraction,
    )
    bit_logits = detector.decode_bits(features, weights)
    return presence_logits, bit_logits, weights


@torch.no_grad()
def present_from_presence_logits(
    presence_logits: torch.Tensor,
    threshold: float,
    min_positive_fraction: float,
) -> torch.Tensor:
    prob = torch.sigmoid(presence_logits)
    return ((prob >= float(threshold)).float().mean(dim=-1) > float(min_positive_fraction))


@torch.no_grad()
def predicted_binary_mask(
    presence_logits: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    return (torch.sigmoid(presence_logits) >= float(threshold)).float()


def embed_batch(
    x: torch.Tensor,
    bits: torch.Tensor,
    model: WatermarkNet,
    cfg: Dict[str, Any],
    sample_rate: int,
    watermark_len: int,
    p_embed: float,
    embed_mode_override: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    embed_cfg = cfg.get("embed", {}) or {}
    embed_mode = str(embed_mode_override or embed_cfg.get("mode", "random_window")).lower()
    window_start_mode = str(embed_cfg.get("start_mode", "random")).lower()

    if embed_mode in ("full", "all"):
        do_embed = torch.rand((x.shape[0],), device=x.device) < float(p_embed)
        return embed_full(x=x, bits=bits, encoder=model.encoder, do_embed=do_embed)

    if embed_mode in ("dropout", "dropout_holes", "holes"):
        nh = embed_cfg.get("num_holes", (0, 3))
        hs = embed_cfg.get("hole_seconds", (0.25, 1.0))
        return embed_dropout_holes(
            x=x,
            bits=bits,
            encoder=model.encoder,
            sample_rate=sample_rate,
            p_embed=float(p_embed),
            ensure_prefix_seconds=float(embed_cfg.get("ensure_prefix_seconds", 0.5)),
            num_holes=(int(nh[0]), int(nh[1])),
            hole_seconds=(float(hs[0]), float(hs[1])),
            ramp_ms=float(embed_cfg.get("ramp_ms", 10.0)),
        )

    return embed_random_window(
        x=x,
        bits=bits,
        encoder=model.encoder,
        watermark_len=watermark_len,
        p_embed=float(p_embed),
        start_mode=window_start_mode,
    )
