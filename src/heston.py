"""Stage 5a: Heston simulation producing a synthetic "pairs" table in the same schema.

Under Heston the index and its variance follow

    dS = mu S dt + sqrt(v) S dW_1
    dv = kappa (theta - v) dt + xi sqrt(v) dW_2,     corr(dW_1, dW_2) = rho

Because implied volatility moves with S (rho < 0 gives the equity skew), the
minimum-variance hedge is not the Black-Scholes delta.  To first order

    delta_MV = delta_BS + vega_BS * d(sigma_imp)/dS

so on simulated data we know the *true* correction and can test whether the
residual network recovers it.  That is the point of this module.

Option prices use the semi-analytic Heston characteristic-function formula
(Gauss-Legendre quadrature); implied vols are backed out with ``bs.implied_vol``
so the simulated table carries the same features as the real one.

Simulating a full chain every day is slow, so by default we price a modest grid
(e.g. 15 strikes x 4 maturities) on each of ~1,000 days.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bs import bs_delta, bs_gamma, bs_vega, implied_vol


@dataclass
class HestonParams:
    kappa: float = 2.0     # mean-reversion speed of variance
    theta: float = 0.04    # long-run variance (20% vol)
    xi: float = 0.5        # vol of vol
    rho: float = -0.7      # leverage correlation
    v0: float = 0.04       # initial variance
    r: float = 0.01        # risk-free rate
    q: float = 0.0         # dividend yield
    mu: float = 0.05       # real-world drift (only matters for the path)


# --------------------------------------------------------------------------- #
# Pricing                                                                     #
# --------------------------------------------------------------------------- #
def _cf(u, S, T, p: HestonParams, v):
    """Heston characteristic function of log S_T (Gatheral / Albrecher 'little trap' form)."""
    a = p.kappa * p.theta
    b = p.kappa
    d = np.sqrt((p.rho * p.xi * 1j * u - b) ** 2 + p.xi**2 * (1j * u + u**2))
    g = (b - p.rho * p.xi * 1j * u - d) / (b - p.rho * p.xi * 1j * u + d)
    C = (p.r - p.q) * 1j * u * T + a / p.xi**2 * (
        (b - p.rho * p.xi * 1j * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = (b - p.rho * p.xi * 1j * u - d) / p.xi**2 * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    return np.exp(C + D * v + 1j * u * np.log(S))


def heston_call(S, K, T, p: HestonParams, v, n_quad: int = 256, u_max: float = 200.0):
    """European call price by Fourier inversion (Gauss-Legendre on [0, u_max])."""
    S, K, T, v = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (S, K, T, v))
    x, w = np.polynomial.legendre.leggauss(n_quad)
    u = 0.5 * u_max * (x + 1); w = 0.5 * u_max * w
    u = u[None, :]; w = w[None, :]
    S_, K_, T_, v_ = S[:, None], K[:, None], T[:, None], v[:, None]
    cf1 = _cf(u - 1j, S_, T_, p, v_)
    cf2 = _cf(u, S_, T_, p, v_)
    F = S_ * np.exp((p.r - p.q) * T_)
    P1 = 0.5 + (w * np.real(np.exp(-1j * u * np.log(K_)) * cf1 / (1j * u * F))).sum(1) / np.pi
    P2 = 0.5 + (w * np.real(np.exp(-1j * u * np.log(K_)) * cf2 / (1j * u))).sum(1) / np.pi
    return np.exp(-p.q * T[:, None].ravel()) * S * P1 - K * np.exp(-p.r * T) * P2


def heston_put(S, K, T, p: HestonParams, v):
    c = heston_call(S, K, T, p, v)
    return c - S * np.exp(-p.q * T) + K * np.exp(-p.r * T)


# --------------------------------------------------------------------------- #
# Path simulation                                                             #
# --------------------------------------------------------------------------- #
def simulate_path(p: HestonParams, n_days: int, S0: float = 3000.0, dt: float = 1 / 252,
                  seed: int = 0) -> pd.DataFrame:
    """One real-world path of (S, v) with daily steps (full-truncation Euler)."""
    rng = np.random.default_rng(seed)
    S, v = np.empty(n_days + 1), np.empty(n_days + 1)
    S[0], v[0] = S0, p.v0
    for t in range(n_days):
        z1 = rng.standard_normal(); z2 = p.rho * z1 + np.sqrt(1 - p.rho**2) * rng.standard_normal()
        vp = max(v[t], 0.0)
        S[t + 1] = S[t] * np.exp((p.mu - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1)
        v[t + 1] = v[t] + p.kappa * (p.theta - vp) * dt + p.xi * np.sqrt(vp * dt) * z2
    dates = pd.bdate_range("2016-01-04", periods=n_days + 1)
    return pd.DataFrame({"date": dates, "S": S, "v": v})


def simulate_chain(p: HestonParams = HestonParams(), n_days: int = 1000, seed: int = 0,
                   moneyness_grid=np.linspace(0.85, 1.15, 15), maturities_days=(30, 60, 91, 182),
                   kinds=("C", "P")) -> pd.DataFrame:
    """Daily option chain on a Heston path, in the standard long schema of ``src.load``.

    Strikes are fixed at day 0 for each maturity bucket and rolled every ``maturity`` days
    so contracts persist from one day to the next (needed for pairing).
    """
    path = simulate_path(p, n_days, seed=seed)
    rows = []
    for m_days in maturities_days:
        # roll a new set of strikes each time the previous set expires
        for start in range(0, n_days, m_days):
            S_start = path["S"].iloc[start]
            strikes = np.round(S_start * moneyness_grid / 5) * 5
            expiry = path["date"].iloc[min(start + m_days, n_days)]
            for t in range(start, min(start + m_days, n_days + 1)):
                S, v, date = path["S"].iloc[t], path["v"].iloc[t], path["date"].iloc[t]
                dte = (expiry - date).days
                if dte <= 0:
                    continue
                T = dte / 365.0
                K = strikes
                call = heston_call(np.full_like(K, S), K, np.full_like(K, T), p, np.full_like(K, v))
                put = call - S * np.exp(-p.q * T) + K * np.exp(-p.r * T)
                for kind, price in zip(kinds, (call, put)):
                    rows.append(pd.DataFrame({
                        "date": date, "expiry": expiry, "type": kind, "strike": K, "S": S,
                        "dte": float(dte), "T": T, "mid": price, "bid": price, "ask": price,
                        "last": price, "volume": 1.0, "bid_size": 1.0, "ask_size": 1.0,
                        "v_true": v,
                    }))
    df = pd.concat(rows, ignore_index=True)
    df["moneyness"] = df["S"] / df["strike"]
    df["iv"] = implied_vol(df["mid"], df["S"], df["strike"], df["T"], p.r, df["type"].to_numpy(), p.q)
    df = df.dropna(subset=["iv"])
    df["delta"] = bs_delta(df["S"], df["strike"], df["T"], p.r, df["iv"], df["type"].to_numpy(), p.q)
    df["vega"] = bs_vega(df["S"], df["strike"], df["T"], p.r, df["iv"], p.q)
    df["gamma"] = bs_gamma(df["S"], df["strike"], df["T"], p.r, df["iv"], p.q)
    df["theta"] = np.nan   # not needed for the simulation study; fill if used as a feature
    df["rho"] = np.nan
    return df


def true_mv_correction(df: pd.DataFrame, p: HestonParams, bump: float = 1e-3) -> np.ndarray:
    """Numerical  vega_BS * d(sigma_imp)/dS  from the Heston pricer: the 'right answer'
    the residual network should approximate.  d(sigma_imp)/dS is obtained by repricing
    at S(1+bump) with v moved along the leverage correlation."""
    S, K, T, v = df["S"].to_numpy(), df["strike"].to_numpy(), df["T"].to_numpy(), df["v_true"].to_numpy()
    kinds = df["type"].to_numpy()
    dS = S * bump
    # conditional expected variance change given dS (Heston regression coefficient)
    dv = p.rho * p.xi * np.sqrt(v) * dS / (S * np.sqrt(v))  # = rho * xi * dS / S
    S_up, v_up = S + dS, np.maximum(v + dv, 1e-8)
    price_up = np.where(kinds == "C", heston_call(S_up, K, T, p, v_up), heston_put(S_up, K, T, p, v_up))
    iv_up = implied_vol(price_up, S_up, K, T, p.r, kinds, p.q)
    d_iv_dS = (iv_up - df["iv"].to_numpy()) / dS
    return df["vega"].to_numpy() * d_iv_dS
