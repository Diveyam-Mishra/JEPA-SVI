from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jepa_iv.black_scholes import black_scholes_price, greeks


@dataclass(frozen=True)
class HedgeBacktestResult:
    pnl: np.ndarray
    pnl_variance: float
    mean_turnover: float
    net_pnl: np.ndarray


def delta_from_surface(
    option_type: str,
    spot: float,
    strike: float,
    tenor: float,
    rate: float,
    implied_vol: float,
) -> float:
    value = greeks(option_type, spot, strike, tenor, rate, implied_vol).delta
    return float(np.clip(value, -1.0, 1.0))


def delta_hedging_backtest(
    option_type: str,
    spots: np.ndarray,
    strikes: np.ndarray,
    tenors: np.ndarray,
    predicted_iv: np.ndarray,
    realised_iv: np.ndarray,
    *,
    rate: float = 0.05,
    transaction_cost_bps: float = 0.0,
) -> HedgeBacktestResult:
    """Daily delta hedge a one-option portfolio over aligned observations."""

    spots = np.asarray(spots, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    tenors = np.asarray(tenors, dtype=float)
    predicted_iv = np.asarray(predicted_iv, dtype=float)
    realised_iv = np.asarray(realised_iv, dtype=float)
    n = len(spots)
    if min(map(len, [spots, strikes, tenors, predicted_iv, realised_iv])) != n:
        raise ValueError("all input series must have the same length")
    if n < 2:
        raise ValueError("at least two observations are required")

    deltas = np.asarray(
        [
            delta_from_surface(option_type, spots[i], strikes[i], max(tenors[i], 1 / 365), rate, predicted_iv[i])
            for i in range(n)
        ]
    )
    option_values = np.asarray(
        [
            black_scholes_price(option_type, spots[i], strikes[i], max(tenors[i], 1 / 365), rate, realised_iv[i])
            for i in range(n)
        ]
    )
    option_pnl = np.diff(option_values)
    hedge_pnl = -deltas[:-1] * np.diff(spots)
    turnover = np.abs(np.diff(deltas))
    costs = transaction_cost_bps / 10_000.0 * turnover * spots[1:]
    pnl = option_pnl + hedge_pnl
    net_pnl = pnl - costs
    return HedgeBacktestResult(
        pnl=pnl,
        pnl_variance=float(np.var(pnl, ddof=1)),
        mean_turnover=float(np.mean(turnover)),
        net_pnl=net_pnl,
    )
