from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 7, stride: int = 1, groups: int = 8):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad)
        gn_groups = min(groups, out_ch)
        self.gn = nn.GroupNorm(gn_groups, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class Encoder1D(nn.Module):
    """
    Encoder that produces an additive residual conditioned on message bits.
    """
    def __init__(self, nbits: int = 32, msg_dim: int = 32, hidden: int = 64, max_delta: float = 0.02):
        super().__init__()
        self.nbits = int(nbits)
        self.msg_dim = int(msg_dim)
        self.hidden = int(hidden)
        self.max_delta = float(max_delta)

        self.msg_fc = nn.Sequential(
            nn.Linear(self.nbits, self.msg_dim),
            nn.GELU(),
            nn.Linear(self.msg_dim, self.msg_dim),
        )

        in_ch = 1 + self.msg_dim
        self.net = nn.Sequential(
            ConvBlock(in_ch, hidden, k=7, stride=1),
            ConvBlock(hidden, hidden, k=7, stride=1),
            ConvBlock(hidden, hidden, k=7, stride=1),
            nn.Conv1d(hidden, 1, kernel_size=7, padding=3),
        )

    def forward(self, x: torch.Tensor, bits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, 1, T]
        bits: [B, nbits] in {0,1}
        returns (y, residual)
        """
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected x [B,1,T], got {x.shape}")
        if bits.ndim != 2 or bits.shape[1] != self.nbits:
            raise ValueError(f"Expected bits [B,{self.nbits}], got {bits.shape}")

        B, _, T = x.shape
        msg = self.msg_fc(bits).unsqueeze(-1).expand(B, self.msg_dim, T)
        inp = torch.cat([x, msg], dim=1)
        r = self.net(inp)
        residual = torch.tanh(r) * self.max_delta
        y = torch.clamp(x + residual, -1.0, 1.0)
        return y, residual


class Detector1D(nn.Module):
    """
    Detector that outputs:
      - presence logits per frame: [B, T']
      - bit logits per frame: [B, nbits, T']
    """
    def __init__(
        self,
        nbits: int = 32,
        channels: List[int] | None = None,
        kernel_size: int = 7,
        strides: List[int] | None = None,
    ):
        super().__init__()
        self.nbits = int(nbits)
        channels = channels or [32, 64, 128, 128]
        strides = strides or [2, 2, 2, 2]
        if len(channels) != len(strides):
            raise ValueError("channels and strides must have same length")

        layers: List[nn.Module] = []
        in_ch = 1
        self.downsample_factor = 1
        for out_ch, st in zip(channels, strides):
            layers.append(ConvBlock(in_ch, out_ch, k=kernel_size, stride=st))
            in_ch = out_ch
            self.downsample_factor *= int(st)

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv1d(in_ch, 1 + self.nbits, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected x [B,1,T], got {x.shape}")
        z = self.backbone(x)
        logits = self.head(z)  # [B, 1+nbits, T']
        presence = logits[:, 0, :]          # [B, T']
        bit_logits = logits[:, 1:, :]       # [B, nbits, T']
        return presence, bit_logits


class WatermarkNet(nn.Module):
    """
    Convenience wrapper: encoder + detector.
    """
    def __init__(self, encoder: Encoder1D, detector: Detector1D):
        super().__init__()
        self.encoder = encoder
        self.detector = detector

    def forward(self, x: torch.Tensor, bits: torch.Tensor):
        y, residual = self.encoder(x, bits)
        presence, bit_logits = self.detector(y)
        return {
            "y": y,
            "residual": residual,
            "presence": presence,
            "bit_logits": bit_logits,
        }
