from __future__ import annotations

import numpy as np
from statsmodels.tsa.api import VAR


class LatentVARDynamics:
    def __init__(self, maxlags: int = 5) -> None:
        self.maxlags = maxlags

    def fit(self, latents: np.ndarray) -> "LatentVARDynamics":
        if latents.ndim != 2:
            raise ValueError("latents must be 2D: observations x dimensions")
        maxlags = min(self.maxlags, max(1, len(latents) // 5))
        self.model_ = VAR(latents).fit(maxlags=maxlags, ic="aic")
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if steps < 1:
            raise ValueError("steps must be positive")
        lagged = self.model_.endog[-self.model_.k_ar :]
        return self.model_.forecast(lagged, steps=steps)
