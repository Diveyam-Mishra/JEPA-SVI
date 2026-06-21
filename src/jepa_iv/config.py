from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SurfaceGridConfig:
    """Fixed implied-volatility surface grid."""

    moneyness_min: float = 0.8
    moneyness_max: float = 1.2
    moneyness_points: int = 20
    tenor_days: tuple[int, ...] = (7, 14, 21, 30, 45, 60, 90, 120, 180, 240, 300, 365)

    @property
    def moneyness_grid(self) -> np.ndarray:
        return np.linspace(self.moneyness_min, self.moneyness_max, self.moneyness_points)

    @property
    def tenor_years(self) -> np.ndarray:
        return np.asarray(self.tenor_days, dtype=float) / 365.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.moneyness_points, len(self.tenor_days)


@dataclass(frozen=True)
class DataConfig:
    volume_min: int = 10
    open_interest_min: int = 50
    moneyness_bounds: tuple[float, float] = (0.7, 1.3)
    iv_bounds: tuple[float, float] = (0.01, 5.0)
    spread_sigma_cutoff: float = 4.0
    min_retention_ratio: float = 0.60
    risk_free_rate: float = 0.05
    raw_store: Path = Path("data/raw/options.parquet")
    surface_store: Path = Path("data/processed/surfaces.npz")


@dataclass(frozen=True)
class JEPAConfig:
    surface_shape: tuple[int, int] = (20, 12)
    patch_shape: tuple[int, int] = (4, 3)
    embed_dim: int = 128
    encoder_depth: int = 4
    encoder_heads: int = 4
    predictor_depth: int = 2
    mlp_ratio: float = 2.0
    mask_ratio: float = 0.60
    ema_tau: float = 0.996
    ema_tau_final: float = 1.0
    dropout: float = 0.0

    def validate(self) -> None:
        height, width = self.surface_shape
        patch_h, patch_w = self.patch_shape
        if height % patch_h or width % patch_w:
            raise ValueError("surface_shape must be divisible by patch_shape")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")

    @property
    def num_patches(self) -> int:
        self.validate()
        return (self.surface_shape[0] // self.patch_shape[0]) * (
            self.surface_shape[1] // self.patch_shape[1]
        )

    @property
    def patch_dim(self) -> int:
        return self.patch_shape[0] * self.patch_shape[1]


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    checkpoint_every: int = 20
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    log_every: int = 100
    collapse_std_floor: float = 0.01
    effective_rank_floor: float = 0.50
    output_dir: Path = field(default_factory=lambda: Path("runs/jepa"))
