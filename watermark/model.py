from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 8,
    ) -> None:
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
        )
        gn_groups = max(1, min(groups, out_ch))
        while out_ch % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.norm = nn.GroupNorm(gn_groups, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConditionedDilatedResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        msg_dim: int,
        kernel_size: int = 7,
        dilation: int = 1,
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pad = (kernel_size // 2) * dilation
        gn_groups = max(1, min(groups, channels))
        while channels % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1

        self.norm1 = nn.GroupNorm(gn_groups, channels)
        self.conv_dw = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation, groups=channels)
        self.norm2 = nn.GroupNorm(gn_groups, channels)
        self.conv_pw = nn.Conv1d(channels, channels, kernel_size=1)
        self.msg_affine = nn.Linear(msg_dim, channels * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.msg_affine(cond).chunk(2, dim=-1)
        y = self.norm1(x)
        y = y * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        y = self.act(y)
        y = self.conv_dw(y)
        y = self.norm2(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.conv_pw(y)
        return x + y


class DownsampleResBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 9,
        stride: int = 2,
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        gn_groups = max(1, min(groups, out_ch))
        while out_ch % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad)
        self.norm1 = nn.GroupNorm(gn_groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(gn_groups, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.skip(x)
        y = self.act(self.norm1(self.conv1(x)))
        y = self.dropout(y)
        y = self.norm2(self.conv2(y))
        return self.act(y + s)


class Encoder1D(nn.Module):
    """
    Improved encoder:
      - much larger receptive field via dilated residual blocks
      - message conditioning in every block
      - bounded additive residual for better imperceptibility control
    """
    def __init__(
        self,
        nbits: int = 16,
        msg_dim: int = 128,
        hidden: int = 128,
        max_delta: float = 0.06,
        kernel_size: int = 7,
        dilations: Sequence[int] | None = None,
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.nbits = int(nbits)
        self.msg_dim = int(msg_dim)
        self.hidden = int(hidden)
        self.max_delta = float(max_delta)
        self.kernel_size = int(kernel_size)
        self.dilations = list(dilations or [1, 2, 4, 8, 16, 32, 64])

        self.msg_fc = nn.Sequential(
            nn.Linear(self.nbits, self.msg_dim),
            nn.GELU(),
            nn.Linear(self.msg_dim, self.msg_dim),
        )
        self.in_proj = nn.Conv1d(1, self.hidden, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        self.blocks = nn.ModuleList(
            [
                ConditionedDilatedResBlock(
                    channels=self.hidden,
                    msg_dim=self.msg_dim,
                    kernel_size=self.kernel_size,
                    dilation=int(d),
                    groups=groups,
                    dropout=dropout,
                )
                for d in self.dilations
            ]
        )
        gn_groups = max(1, min(groups, self.hidden))
        while self.hidden % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.out_norm = nn.GroupNorm(gn_groups, self.hidden)
        self.out_conv = nn.Conv1d(self.hidden, 1, kernel_size=1)

    def receptive_field(self) -> int:
        rf = self.kernel_size
        for d in self.dilations:
            rf += (self.kernel_size - 1) * int(d)
        return int(rf)

    def forward(self, x: torch.Tensor, bits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected x [B,1,T], got {x.shape}")
        if bits.ndim != 2 or bits.shape[1] != self.nbits:
            raise ValueError(f"Expected bits [B,{self.nbits}], got {bits.shape}")

        _B, _, _T = x.shape
        msg = self.msg_fc(bits)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, msg)
        h = F.gelu(self.out_norm(h))
        residual = torch.tanh(self.out_conv(h)) * self.max_delta
        y = torch.clamp(x + residual, -1.0, 1.0)
        return y, residual


class Detector1D(nn.Module):
    """
    Improved detector:
      - frame-level presence head for localization
      - global masked pooling bit decoder for stable message recovery
      - mean+std pooled statistics are much easier to learn than per-frame bit logits
    """
    def __init__(
        self,
        nbits: int = 16,
        channels: List[int] | None = None,
        kernel_size: int = 9,
        strides: List[int] | None = None,
        bit_mlp_hidden: int = 256,
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.nbits = int(nbits)
        self.channels = list(channels or [64, 128, 256, 256])
        self.strides = list(strides or [2, 2, 2, 2])
        if len(self.channels) != len(self.strides):
            raise ValueError("channels and strides must have same length")

        layers = []
        in_ch = 1
        self.downsample_factor = 1
        for out_ch, st in zip(self.channels, self.strides):
            layers.append(
                DownsampleResBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    stride=int(st),
                    groups=groups,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
            self.downsample_factor *= int(st)

        self.backbone = nn.Sequential(*layers)
        self.presence_head = nn.Conv1d(in_ch, 1, kernel_size=1)
        self.bit_context = nn.Sequential(
            ConvNormAct(in_ch, in_ch, kernel_size=5, stride=1, groups=groups),
            ConvNormAct(in_ch, in_ch, kernel_size=5, stride=1, groups=groups),
        )
        self.bit_mlp = nn.Sequential(
            nn.Linear(in_ch * 2, bit_mlp_hidden),
            nn.GELU(),
            nn.Linear(bit_mlp_hidden, bit_mlp_hidden),
            nn.GELU(),
            nn.Linear(bit_mlp_hidden, self.nbits),
        )

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected x [B,1,T], got {x.shape}")
        z = self.backbone(x)
        presence_logits = self.presence_head(z).squeeze(1)
        return presence_logits, z

    def masked_stats_pool(self, z: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
        z = self.bit_context(z)
        if weights is None:
            mean = z.mean(dim=-1)
            std = z.std(dim=-1, unbiased=False)
            return torch.cat([mean, std], dim=1)

        w = weights.clamp(0.0, 1.0)
        wsum = w.sum(dim=-1, keepdim=True)
        fallback = (wsum <= 1e-4).float()
        wsum = wsum.clamp_min(1e-4)

        mean = (z * w.unsqueeze(1)).sum(dim=-1) / wsum
        var = (((z - mean.unsqueeze(-1)) ** 2) * w.unsqueeze(1)).sum(dim=-1) / wsum
        std = torch.sqrt(var + 1e-5)

        if float(fallback.max().item()) > 0.0:
            g_mean = z.mean(dim=-1)
            g_std = z.std(dim=-1, unbiased=False)
            mean = mean * (1.0 - fallback) + g_mean * fallback
            std = std * (1.0 - fallback) + g_std * fallback

        return torch.cat([mean, std], dim=1)

    def decode_bits(self, features: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self.masked_stats_pool(features, weights)
        return self.bit_mlp(pooled)

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        presence_logits, features = self.forward_features(x)
        bit_logits = self.decode_bits(features, weights)
        return presence_logits, bit_logits


class WatermarkNet(nn.Module):
    def __init__(self, encoder: Encoder1D, detector: Detector1D):
        super().__init__()
        self.encoder = encoder
        self.detector = detector

    def forward(self, x: torch.Tensor, bits: torch.Tensor, weights: torch.Tensor | None = None):
        y, residual = self.encoder(x, bits)
        presence, bit_logits = self.detector(y, weights=weights)
        return {
            "y": y,
            "residual": residual,
            "presence": presence,
            "bit_logits": bit_logits,
        }
