from __future__ import annotations

import numpy as np
from scipy import stats


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    true_var = np.maximum(np.asarray(y_true) ** 2, eps)
    pred_var = np.maximum(np.asarray(y_pred) ** 2, eps)
    return float(np.mean(true_var / pred_var + np.log(pred_var)))


def effective_rank(representations: np.ndarray, eps: float = 1e-12) -> float:
    singular = np.linalg.svd(representations - representations.mean(axis=0), compute_uv=False)
    if singular.sum() <= eps:
        return 0.0
    probs = singular / singular.sum()
    entropy = -np.sum(probs * np.log(probs + eps))
    return float(np.exp(entropy) / len(singular))


def diebold_mariano(
    actual: np.ndarray,
    forecast_a: np.ndarray,
    forecast_b: np.ndarray,
    *,
    power: int = 2,
) -> tuple[float, float]:
    loss_diff = np.mean(np.abs(actual - forecast_a) ** power - np.abs(actual - forecast_b) ** power, axis=tuple(range(1, actual.ndim)))
    if len(loss_diff) < 2:
        raise ValueError("Diebold-Mariano test needs at least two forecast periods")
    dm_stat = float(loss_diff.mean() / (loss_diff.std(ddof=1) / np.sqrt(len(loss_diff))))
    p_value = float(2 * (1 - stats.t.cdf(abs(dm_stat), df=len(loss_diff) - 1)))
    return dm_stat, p_value
