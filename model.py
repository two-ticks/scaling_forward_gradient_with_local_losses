"""LocalMixer architecture from Ren et al. 2023 (Scaling Forward Gradient With Local Losses).

The grouped channel design is what makes the architecture compatible with local
learning: each group is a small parallel sub-network, so a per-group local loss
sees only a fraction of the parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _uniform_(t, fan_in):
    bound = 1.0 / math.sqrt(max(fan_in, 1))
    nn.init.uniform_(t, -bound, bound)


class GroupedLinear(nn.Module):
    """Per-group linear: (..., G, in) -> (..., G, out), separate weights per group."""

    def __init__(self, num_groups: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_groups, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(num_groups, out_dim))
        _uniform_(self.weight, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...gi,gio->...go", x, self.weight) + self.bias


class TokenMixer(nn.Module):
    """Mixes patches per group along the spatial axis P.

    Input/output: (B, P, G, C). Hidden weight has shape (G, P, H) and (G, H, P);
    each group has its own MLP over patches. This is the LocalMixer "linear token
    mixing layer" used in place of attention.
    """

    def __init__(self, num_groups: int, num_patches: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Parameter(torch.empty(num_groups, num_patches, hidden))
        self.fc2 = nn.Parameter(torch.empty(num_groups, hidden, num_patches))
        self.b1 = nn.Parameter(torch.zeros(num_groups, hidden))
        self.b2 = nn.Parameter(torch.zeros(num_groups, num_patches))
        _uniform_(self.fc1, num_patches)
        _uniform_(self.fc2, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.permute(0, 3, 2, 1)  # (B, C, G, P)
        h = torch.einsum("bcgp,gph->bcgh", x_t, self.fc1) + self.b1
        h = F.gelu(h)
        out = torch.einsum("bcgh,ghp->bcgp", h, self.fc2) + self.b2
        return out.permute(0, 3, 2, 1)  # (B, P, G, C)


class ChannelMixer(nn.Module):
    """Per-group channel MLP. Input/output: (B, P, G, C)."""

    def __init__(self, num_groups: int, channels: int, hidden: int):
        super().__init__()
        self.fc1 = GroupedLinear(num_groups, channels, hidden)
        self.fc2 = GroupedLinear(num_groups, hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class GroupLayerNorm(nn.Module):
    """LayerNorm over the C axis with per-group affine parameters."""

    def __init__(self, num_groups: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_groups, channels))
        self.bias = nn.Parameter(torch.zeros(num_groups, channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight + self.bias


class PatchEmbed(nn.Module):
    """Image (B, C_in, H, W) -> tokens (B, P, G, C) via strided conv patchify."""

    def __init__(self, in_channels: int, patch_size: int, num_groups: int, channels: int):
        super().__init__()
        self.num_groups = num_groups
        self.channels = channels
        self.proj = nn.Conv2d(in_channels, num_groups * channels,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, P, G*C)
        return x.view(b, h * w, self.num_groups, self.channels)


class MixerBlock(nn.Module):
    """One LocalMixer block. Block 0 owns the patch embedding so it is a single
    "trainable unit" for forward gradient (its parameters are perturbed together)."""

    def __init__(self, num_groups: int, num_patches: int, channels: int,
                 token_hidden: int, channel_hidden: int,
                 patch_embed: PatchEmbed | None = None):
        super().__init__()
        self.patch_embed = patch_embed
        self.norm1 = GroupLayerNorm(num_groups, channels)
        self.token = TokenMixer(num_groups, num_patches, token_hidden)
        self.norm2 = GroupLayerNorm(num_groups, channels)
        self.chan = ChannelMixer(num_groups, channels, channel_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_embed is not None:
            x = self.patch_embed(x)
        x = x + self.token(self.norm1(x))
        x = x + self.chan(self.norm2(x))
        return x


class LocalHead(nn.Module):
    """Spatially-averaged per-group linear classifier; group logits are averaged.

    This is the local loss head: each block has one. The local loss is
    cross-entropy between this head's prediction and the true label.
    """

    def __init__(self, num_groups: int, channels: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_groups, channels, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        _uniform_(self.weight, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, P, G, C) -> (B, K)
        x = x.mean(dim=1)  # average over patches
        logits = torch.einsum("bgc,gck->bgk", x, self.weight)
        return logits.mean(dim=1) + self.bias


class LocalMixer(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int,
                 num_groups: int, channels: int, token_hidden: int,
                 channel_hidden: int, num_blocks: int, num_classes: int):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        embed = PatchEmbed(in_channels, patch_size, num_groups, channels)
        blocks = []
        for i in range(num_blocks):
            blocks.append(MixerBlock(
                num_groups, num_patches, channels, token_hidden, channel_hidden,
                patch_embed=embed if i == 0 else None,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleList([
            LocalHead(num_groups, channels, num_classes) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Eval forward: average per-block head logits (deep ensemble)."""
        all_logits = []
        for blk, head in zip(self.blocks, self.heads):
            x = blk(x)
            all_logits.append(head(x))
        return torch.stack(all_logits, 0).mean(0)
