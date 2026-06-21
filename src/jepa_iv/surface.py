from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from jepa_iv.black_scholes import black_scholes_price, implied_volatility
from jepa_iv.config import DataConfig, SurfaceGridConfig
from jepa_iv.data import normalize_option_frame


@dataclass(frozen=True)
class SurfaceScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, train_surfaces: np.ndarray) -> "SurfaceScaler":
        mean = np.nanmean(train_surfaces, axis=0)
        std = np.nanstd(train_surfaces, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, surfaces: np.ndarray) -> np.ndarray:
        return (surfaces - self.mean) / self.std

    def inverse_transform(self, surfaces: np.ndarray) -> np.ndarray:
        return surfaces * self.std + self.mean


def add_mid_prices(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_option_frame(frame)
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    out["spread"] = out["ask"] - out["bid"]
    if (out["mid"] < 0).any():
        raise ValueError("negative mid prices detected")
    return out


def compute_iv_columns(frame: pd.DataFrame, config: DataConfig = DataConfig()) -> pd.DataFrame:
    out = add_mid_prices(frame)
    if "dte_days" in out.columns:
        out["time_to_expiry"] = pd.to_numeric(out["dte_days"], errors="coerce") / 365.0
    else:
        out["time_to_expiry"] = (out["expiry"] - out["timestamp"]).dt.total_seconds() / (365.0 * 24 * 3600)
    out = out[out["time_to_expiry"] > 0].copy()
    ivs: list[float] = []
    failures = 0
    for row in out.itertuples(index=False):
        supplied_iv = getattr(row, "iv", np.nan)
        if np.isfinite(supplied_iv) and config.iv_bounds[0] <= supplied_iv <= config.iv_bounds[1]:
            ivs.append(float(supplied_iv))
            continue
        try:
            iv = implied_volatility(
                float(row.mid),
                str(row.option_type),
                float(row.underlying_price),
                float(row.strike),
                float(row.time_to_expiry),
                config.risk_free_rate,
                lower=config.iv_bounds[0],
                upper=config.iv_bounds[1],
            )
        except ValueError:
            failures += 1
            iv = np.nan
        ivs.append(iv)
    out["iv"] = ivs
    out["iv_solver_failed"] = np.isnan(out["iv"])
    out["iv"] = out["iv"].clip(config.iv_bounds[0], config.iv_bounds[1])
    if failures == len(out) and len(out) > 0:
        raise ValueError("IV solver failed for every option row")
    return out


def filter_option_data(frame: pd.DataFrame, config: DataConfig = DataConfig()) -> pd.DataFrame:
    if "iv" not in frame or "spread" not in frame:
        frame = compute_iv_columns(frame, config)
    out = frame.copy()
    out["moneyness"] = out["strike"] / out["underlying_price"]
    spread_mean = out["spread"].mean()
    spread_std = out["spread"].std(ddof=0)
    spread_limit = spread_mean + config.spread_sigma_cutoff * max(spread_std, 1e-12)
    low_m, high_m = config.moneyness_bounds
    mask = (
        (out["volume"] >= config.volume_min)
        & (out["open_interest"] >= config.open_interest_min)
        & (out["spread"] <= spread_limit)
        & out["moneyness"].between(low_m, high_m)
        & out["iv"].between(config.iv_bounds[0], config.iv_bounds[1])
        & (~out["iv_solver_failed"])
    )
    filtered = out.loc[mask].copy()
    if len(out) and len(filtered) / len(out) < config.min_retention_ratio:
        filtered.attrs["retention_warning"] = len(filtered) / len(out)
    return filtered


def interpolate_surface(day_frame: pd.DataFrame, grid: SurfaceGridConfig = SurfaceGridConfig()) -> np.ndarray:
    required = {"moneyness", "time_to_expiry", "iv"}
    missing = required.difference(day_frame.columns)
    if missing:
        raise ValueError(f"surface interpolation missing columns: {sorted(missing)}")
    points = day_frame[["moneyness", "time_to_expiry"]].to_numpy(float)
    values = day_frame["iv"].to_numpy(float)
    target_m, target_t = np.meshgrid(grid.moneyness_grid, grid.tenor_years, indexing="ij")
    target = np.column_stack([target_m.ravel(), target_t.ravel()])
    method = "cubic" if len(day_frame) >= 16 else "linear"
    surface = griddata(points, values, target, method=method)
    if np.isnan(surface).any():
        nearest = griddata(points, values, target, method="nearest")
        surface = np.where(np.isnan(surface), nearest, surface)
    return surface.reshape(grid.shape)


def build_surface_tensor(
    frame: pd.DataFrame,
    grid: SurfaceGridConfig = SurfaceGridConfig(),
    config: DataConfig = DataConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    filtered = filter_option_data(frame, config)
    if filtered.empty:
        raise ValueError("no rows remain after filtering")
    surfaces: list[np.ndarray] = []
    dates: list[np.datetime64] = []
    for ts, day in filtered.groupby(filtered["timestamp"].dt.normalize(), sort=True):
        if len(day) < 8:
            continue
        surfaces.append(interpolate_surface(day, grid))
        dates.append(np.datetime64(ts.date()))
    if not surfaces:
        raise ValueError("no daily surfaces could be constructed")
    return np.stack(surfaces).astype(np.float32), np.asarray(dates)


def butterfly_violation_rate(surface: np.ndarray, spot: float, rate: float, tenors: np.ndarray) -> float:
    strikes = np.linspace(0.8, 1.2, surface.shape[0]) * spot
    # Clamp IVs to a small positive floor — negative/zero IVs from model
    # predictions are physically invalid and counted as structural violations.
    safe_surface = np.maximum(surface, 1e-6)
    violations = 0
    checks = 0
    for j, tenor in enumerate(tenors):
        calls = [
            black_scholes_price("call", spot, float(k), float(tenor), rate, float(safe_surface[i, j]))
            for i, k in enumerate(strikes)
        ]
        for i in range(1, len(calls) - 1):
            violations += int(calls[i - 1] - 2 * calls[i] + calls[i + 1] < -1e-8)
            checks += 1
    return violations / checks if checks else 0.0


def calendar_violation_rate(surface: np.ndarray, tenors: np.ndarray) -> float:
    total_variance = surface**2 * tenors.reshape(1, -1)
    diffs = np.diff(total_variance, axis=1)
    return float(np.mean(diffs < -1e-8))
