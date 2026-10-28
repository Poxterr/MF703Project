"""Black-Scholes prices and Greeks (vectorised).  Used by the Heston simulator,
the interpretability plots, and the sanity check that the vendor's Greeks are consistent.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1d2(S, K, T, r, sigma, q=0.0):
    S, K, T, r, sigma, q = map(np.asarray, (S, K, T, r, sigma, q))
    T = np.maximum(T, 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_price(S, K, T, r, sigma, kind="C", q=0.0):
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    call = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    if np.ndim(kind) == 0:
        return call if kind == "C" else call - S * disc_q + K * disc_r
    is_call = np.asarray(kind) == "C"
    return np.where(is_call, call, call - S * disc_q + K * disc_r)


def bs_delta(S, K, T, r, sigma, kind="C", q=0.0):
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    call = np.exp(-q * T) * norm.cdf(d1)
    if np.ndim(kind) == 0:
        return call if kind == "C" else call - np.exp(-q * T)
    return np.where(np.asarray(kind) == "C", call, call - np.exp(-q * T))


def bs_vega(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def implied_vol(price, S, K, T, r, kind="C", q=0.0, tol=1e-8, max_iter=100):
    """Vectorised bisection for implied volatility.  Returns NaN where no root exists."""
    price, S, K, T = map(np.asarray, (price, S, K, T))
    lo, hi = np.full(price.shape, 1e-4), np.full(price.shape, 5.0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        diff = bs_price(S, K, T, r, mid, kind, q) - price
        lo = np.where(diff < 0, mid, lo)
        hi = np.where(diff >= 0, mid, hi)
        if np.all(hi - lo < tol):
            break
    iv = 0.5 * (lo + hi)
    bad = (price <= bs_price(S, K, T, r, 1e-4, kind, q)) | (price >= bs_price(S, K, T, r, 5.0, kind, q))
    return np.where(bad, np.nan, iv)
