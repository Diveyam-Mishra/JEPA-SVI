from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.a, self.b, self.rho, self.m, self.sigma], dtype=float)


def svi_total_variance(log_moneyness: np.ndarray, params: SVIParams) -> np.ndarray:
    k = np.asarray(log_moneyness, dtype=float)
    return params.a + params.b * (
        params.rho * (k - params.m) + np.sqrt((k - params.m) ** 2 + params.sigma**2)
    )


def _unpack(raw: np.ndarray) -> SVIParams:
    a = raw[0]
    b = np.exp(raw[1])
    rho = np.tanh(raw[2])
    m = raw[3]
    sigma = np.exp(raw[4])
    return SVIParams(float(a), float(b), float(rho), float(m), float(sigma))


def fit_svi_slice(
    log_moneyness: np.ndarray,
    implied_vol: np.ndarray,
    tenor: float,
    *,
    initial: SVIParams | None = None,
) -> SVIParams:
    if tenor <= 0:
        raise ValueError("tenor must be positive")
    k = np.asarray(log_moneyness, dtype=float)
    target = np.asarray(implied_vol, dtype=float) ** 2 * tenor
    valid = np.isfinite(k) & np.isfinite(target) & (target > 0)
    if valid.sum() < 5:
        raise ValueError("at least five valid points are required to fit SVI")
    k = k[valid]
    target = target[valid]
    if initial is None:
        initial = SVIParams(
            a=float(np.percentile(target, 10)),
            b=0.1,
            rho=-0.3,
            m=float(np.median(k)),
            sigma=0.1,
        )
    x0 = np.asarray(
        [initial.a, np.log(initial.b), np.arctanh(np.clip(initial.rho, -0.999, 0.999)), initial.m, np.log(initial.sigma)]
    )

    def objective(raw: np.ndarray) -> float:
        params = _unpack(raw)
        pred = svi_total_variance(k, params)
        penalty = 1e3 * np.mean(np.minimum(pred, 0.0) ** 2)
        return float(np.mean((pred - target) ** 2) + penalty)

    result = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 4000})
    if not result.success:
        result = minimize(objective, result.x, method="BFGS", options={"maxiter": 2000})
    params = _unpack(result.x)
    if np.any(svi_total_variance(k, params) <= 0):
        raise ValueError("SVI fit produced non-positive total variance")
    return params


def evaluate_svi_surface(
    params_by_tenor: list[SVIParams],
    moneyness_grid: np.ndarray,
    tenors: np.ndarray,
) -> np.ndarray:
    if len(params_by_tenor) != len(tenors):
        raise ValueError("one parameter set is required per tenor")
    k = np.log(np.asarray(moneyness_grid, dtype=float))
    surface = np.empty((len(k), len(tenors)), dtype=float)
    for j, (params, tenor) in enumerate(zip(params_by_tenor, tenors, strict=True)):
        total_var = svi_total_variance(k, params)
        surface[:, j] = np.sqrt(np.maximum(total_var / tenor, 1e-12))
    return surface


def fit_svi_surface(surface: np.ndarray, moneyness_grid: np.ndarray, tenors: np.ndarray) -> list[SVIParams]:
    k = np.log(np.asarray(moneyness_grid, dtype=float))
    return [fit_svi_slice(k, surface[:, j], float(tenor)) for j, tenor in enumerate(tenors)]
