from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _validate_inputs(spot: float, strike: float, time_to_expiry: float, sigma: float) -> None:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")


def d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    sigma: float,
) -> tuple[float, float]:
    _validate_inputs(spot, strike, time_to_expiry, sigma)
    vol_sqrt_t = sigma * np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * time_to_expiry) / vol_sqrt_t
    return float(d1), float(d1 - vol_sqrt_t)


def black_scholes_price(
    option_type: str,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    sigma: float,
) -> float:
    d1, d2 = d1_d2(spot, strike, time_to_expiry, rate, sigma)
    discount = np.exp(-rate * time_to_expiry)
    if option_type == "call":
        return float(spot * norm.cdf(d1) - strike * discount * norm.cdf(d2))
    if option_type == "put":
        return float(strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1))
    raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_vega(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    sigma: float,
) -> float:
    d1, _ = d1_d2(spot, strike, time_to_expiry, rate, sigma)
    return float(spot * norm.pdf(d1) * np.sqrt(time_to_expiry))


def greeks(
    option_type: str,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    sigma: float,
) -> Greeks:
    d1, d2 = d1_d2(spot, strike, time_to_expiry, rate, sigma)
    pdf = norm.pdf(d1)
    discount = np.exp(-rate * time_to_expiry)
    gamma = pdf / (spot * sigma * np.sqrt(time_to_expiry))
    vega = spot * pdf * np.sqrt(time_to_expiry)
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = -(spot * pdf * sigma) / (2 * np.sqrt(time_to_expiry)) - rate * strike * discount * norm.cdf(d2)
        rho = strike * time_to_expiry * discount * norm.cdf(d2)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
        theta = -(spot * pdf * sigma) / (2 * np.sqrt(time_to_expiry)) + rate * strike * discount * norm.cdf(-d2)
        rho = -strike * time_to_expiry * discount * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return Greeks(float(delta), float(gamma), float(vega), float(theta), float(rho))


def implied_volatility(
    price: float,
    option_type: str,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    *,
    lower: float = 0.01,
    upper: float = 5.0,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """Solve Black-Scholes implied volatility with Newton and Brent fallback."""

    if price <= 0 or not np.isfinite(price):
        raise ValueError("price must be positive and finite")
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be positive")

    intrinsic = max(0.0, spot - strike * np.exp(-rate * time_to_expiry))
    if option_type == "put":
        intrinsic = max(0.0, strike * np.exp(-rate * time_to_expiry) - spot)
    if price < intrinsic - 1e-10:
        raise ValueError("option price is below discounted intrinsic value")

    seed = np.sqrt(2 * np.pi / time_to_expiry) * price / max(spot, 1e-12)
    sigma = float(np.clip(seed, lower, upper))
    for _ in range(max_iterations):
        model = black_scholes_price(option_type, spot, strike, time_to_expiry, rate, sigma)
        diff = model - price
        if abs(diff) < tolerance:
            return float(np.clip(sigma, lower, upper))
        vega = black_scholes_vega(spot, strike, time_to_expiry, rate, sigma)
        if vega < 1e-10:
            break
        sigma_next = sigma - diff / vega
        if not lower <= sigma_next <= upper or not np.isfinite(sigma_next):
            break
        sigma = float(sigma_next)

    def objective(vol: float) -> float:
        return black_scholes_price(option_type, spot, strike, time_to_expiry, rate, vol) - price

    low_value = objective(lower)
    high_value = objective(upper)
    if low_value * high_value > 0:
        if abs(low_value) < abs(high_value):
            return lower
        return upper
    return float(brentq(objective, lower, upper, xtol=tolerance, rtol=tolerance, maxiter=max_iterations))
