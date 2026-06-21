from __future__ import annotations

import math

import torch
from torch import nn

from jepa_iv.config import JEPAConfig


def patchify(x: torch.Tensor, patch_shape: tuple[int, int]) -> torch.Tensor:
    batch, height, width = x.shape
    patch_h, patch_w = patch_shape
    if height % patch_h or width % patch_w:
        raise ValueError("surface dimensions must be divisible by patch size")
    x = x.reshape(batch, height // patch_h, patch_h, width // patch_w, patch_w)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    return x.reshape(batch, -1, patch_h * patch_w)


def unpatchify(tokens: torch.Tensor, surface_shape: tuple[int, int], patch_shape: tuple[int, int]) -> torch.Tensor:
    batch = tokens.shape[0]
    height, width = surface_shape
    patch_h, patch_w = patch_shape
    tokens = tokens.reshape(batch, height // patch_h, width // patch_w, patch_h, patch_w)
    tokens = tokens.permute(0, 1, 3, 2, 4).contiguous()
    return tokens.reshape(batch, height, width)


def sinusoidal_position_encoding(num_tokens: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(num_tokens, dim)
    position = torch.arange(0, num_tokens, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class SurfaceEncoder(nn.Module):
    def __init__(self, config: JEPAConfig, *, depth: int | None = None) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.patch_embed = nn.Linear(config.patch_dim, config.embed_dim)
        pe = sinusoidal_position_encoding(config.num_patches, config.embed_dim)
        self.register_buffer("position_encoding", pe, persistent=False)
        self.blocks = nn.ModuleList(
            TransformerBlock(config.embed_dim, config.encoder_heads, config.mlp_ratio, config.dropout)
            for _ in range(depth if depth is not None else config.encoder_depth)
        )
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, surfaces: torch.Tensor, token_indices: torch.Tensor | None = None) -> torch.Tensor:
        patches = patchify(surfaces, self.config.patch_shape)
        x = self.patch_embed(patches) + self.position_encoding.to(patches.device)
        if token_indices is not None:
            gather = token_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.gather(x, dim=1, index=gather)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class JEPAPredictor(nn.Module):
    def __init__(self, config: JEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(config.embed_dim, config.encoder_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.predictor_depth)
        )
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
        position_encoding: torch.Tensor,
    ) -> torch.Tensor:
        batch, target_count = target_indices.shape
        target_pos = position_encoding[target_indices].to(context_tokens.device)
        mask_tokens = self.mask_token.expand(batch, target_count, -1) + target_pos
        context_pos = position_encoding[context_indices].to(context_tokens.device)
        x = torch.cat([context_tokens + context_pos, mask_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x[:, -target_count:, :])


class SurfaceJEPA(nn.Module):
    def __init__(self, config: JEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = SurfaceEncoder(config)
        self.target_encoder = SurfaceEncoder(config)
        self.predictor = JEPAPredictor(config)
        self.reset_target_encoder()

    @torch.no_grad()
    def reset_target_encoder(self) -> None:
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update_target_encoder(self, tau: float) -> None:
        for target, context in zip(self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True):
            target.data.mul_(tau).add_(context.data, alpha=1.0 - tau)

    def forward(
        self,
        surfaces: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.context_encoder(surfaces, context_indices)
        pred = self.predictor(
            context,
            context_indices,
            target_indices,
            self.context_encoder.position_encoding,
        )
        with torch.no_grad():
            target = self.target_encoder(surfaces, target_indices)
        return pred, target


class SurfaceDecoder(nn.Module):
    def __init__(self, latent_dim: int, surface_shape: tuple[int, int]) -> None:
        super().__init__()
        output_dim = surface_shape[0] * surface_shape[1]
        self.surface_shape = surface_shape
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, output_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).reshape(latent.shape[0], *self.surface_shape)
