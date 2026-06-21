from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import VAR


@dataclass
class ForecastResult:
    name: str
    prediction: np.ndarray


class RandomWalkBaseline:
    name = "random_walk"

    def predict(self, history: np.ndarray, horizon: int = 1) -> np.ndarray:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        return history[:-horizon]


class HistoricalMeanBaseline:
    name = "historical_mean"

    def fit(self, train: np.ndarray) -> "HistoricalMeanBaseline":
        self.mean_ = train.mean(axis=0)
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        return np.repeat(self.mean_[None, ...], n_periods, axis=0)


class PCAVARBaseline:
    name = "pca_var"

    def __init__(self, n_components: int = 5, maxlags: int = 5) -> None:
        self.n_components = n_components
        self.maxlags = maxlags
        self.pca = PCA(n_components=n_components)

    def fit(self, train: np.ndarray) -> "PCAVARBaseline":
        flat = train.reshape(len(train), -1)
        self.surface_shape_ = train.shape[1:]
        self.pca.fit(flat)
        scores = self.pca.transform(flat)
        self.var_ = VAR(scores).fit(maxlags=min(self.maxlags, max(1, len(scores) // 5)), ic="aic")
        return self

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    def forecast(self, steps: int) -> np.ndarray:
        lagged = self.var_.endog[-self.var_.k_ar :]
        score_forecast = self.var_.forecast(lagged, steps=steps)
        reconstructed = self.pca.inverse_transform(score_forecast)
        return reconstructed.reshape((steps, *self.surface_shape_))


class NelsonSiegelTermStructure:
    """Simple dynamic Nelson-Siegel baseline for ATM term structure."""

    def __init__(self, decay: float = 0.0609) -> None:
        self.decay = decay

    def _loadings(self, tenors: np.ndarray) -> np.ndarray:
        lam_t = np.maximum(self.decay * tenors, 1e-8)
        slope = (1 - np.exp(-lam_t)) / lam_t
        curvature = slope - np.exp(-lam_t)
        return np.column_stack([np.ones_like(tenors), slope, curvature])

    def fit_daily_factors(self, surfaces: np.ndarray, tenors: np.ndarray) -> np.ndarray:
        atm_idx = surfaces.shape[1] // 2
        y = surfaces[:, atm_idx, :]
        x = self._loadings(tenors)
        factors = []
        for row in y:
            factors.append(LinearRegression(fit_intercept=False).fit(x, row).coef_)
        return np.asarray(factors)

    def fit(self, train: np.ndarray, tenors: np.ndarray) -> "NelsonSiegelTermStructure":
        self.tenors_ = tenors
        self.surface_shape_ = train.shape[1:]
        self.factors_ = self.fit_daily_factors(train, tenors)
        self.ar_ = VAR(self.factors_).fit(maxlags=1)
        return self

    def forecast_term_structure(self, steps: int) -> np.ndarray:
        factors = self.ar_.forecast(self.ar_.endog[-self.ar_.k_ar :], steps=steps)
        return factors @ self._loadings(self.tenors_).T
