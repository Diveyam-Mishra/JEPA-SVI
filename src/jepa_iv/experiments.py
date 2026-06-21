from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jepa_iv.metrics import diebold_mariano, mse, qlike


@dataclass(frozen=True)
class MethodScores:
    method: str
    mse: float
    qlike: float


def score_forecasts(actual: np.ndarray, forecasts: dict[str, np.ndarray]) -> list[MethodScores]:
    return [
        MethodScores(method=name, mse=mse(actual, prediction), qlike=qlike(actual, prediction))
        for name, prediction in forecasts.items()
    ]


def region_masks(surface_shape: tuple[int, int]) -> dict[str, tuple[slice | np.ndarray, slice | np.ndarray]]:
    height, width = surface_shape
    m = np.linspace(0.8, 1.2, height)
    return {
        "atm": (np.abs(m - 1.0) <= 0.03, slice(None)),
        "otm_puts": (m < 0.95, slice(None)),
        "otm_calls": (m > 1.05, slice(None)),
        "short_tenor": (slice(None), np.arange(width) < width // 3),
        "long_tenor": (slice(None), np.arange(width) >= 2 * width // 3),
    }


def score_by_region(actual: np.ndarray, prediction: np.ndarray) -> dict[str, MethodScores]:
    scores = {}
    for name, mask in region_masks(actual.shape[1:]).items():
        scores[name] = MethodScores(name, mse(actual[:, mask[0], :][:, :, mask[1]], prediction[:, mask[0], :][:, :, mask[1]]), qlike(actual[:, mask[0], :][:, :, mask[1]], prediction[:, mask[0], :][:, :, mask[1]]))
    return scores


def compare_to_best_baseline(
    actual: np.ndarray,
    jepa_forecast: np.ndarray,
    baseline_forecasts: dict[str, np.ndarray],
) -> tuple[str, float, float]:
    best_name = min(baseline_forecasts, key=lambda name: mse(actual, baseline_forecasts[name]))
    dm, p = diebold_mariano(actual, jepa_forecast, baseline_forecasts[best_name])
    return best_name, dm, p
