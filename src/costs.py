"""Stage 5b: multi-day hedging simulation with transaction costs.

The one-day loss used for training has no rebalancing in it.  To judge a hedge net of costs we
follow each contract over a **hedging episode**: a run of consecutive trading days on which the
contract is in the sample.  A contract that leaves the sample (a missing vendor day, or its delta
moving outside the filter range) and comes back starts a new episode, so a hedge is never carried
across days we have no prices for.

Within an episode, for a short option hedged with the index:

    day t:   hold -1 option, +delta_t units of index
    P&L_t  = -(V_{t+1} - V_t) + delta_t (S_{t+1} - S_t) - c * |delta_t - delta_{t-1}| * S_t

with proportional cost ``c`` per unit of index notional traded (1 bp = 1e-4).  The first day pays
``c * |delta_0| * S_0`` to set the hedge up and the last day pays ``c * |delta_T| * S_{T+1}`` to
take it off, so every model is charged for a complete round trip.  Financing is ignored (one day
of interest on the hedge is two orders of magnitude smaller than the hedging error).

Everything is vectorised: the cost is linear in ``c``, so gross P&L and traded notional are
computed once and each cost level is a subtraction.

Main entry points
-----------------
``hedge_pnl``    one row per episode and model: total P&L, cost, turnover, rebalancing, daily variance
``summarise``    per model: mean and spread of episode P&L, mean cost, turnover
``cost_sweep``   ``summarise`` at several cost levels
``net_gain``     the gain ratio of notebooks 02-03 recomputed on net-of-cost daily P&L
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import KEY


def _delta_col(model: str) -> str:
    return "delta" if model in ("bs", "delta") else (model if model.startswith("delta_") else f"delta_{model}")


def sort_for_episodes(pairs: pd.DataFrame) -> pd.DataFrame:
    """Rows ordered by contract and date, with an integer ``episode`` column.

    A new episode starts when the contract changes or when the previous row's ``date_next`` is not
    this row's ``date`` (the contract was out of the sample in between)."""
    if "date_next" not in pairs:
        raise KeyError("pairs needs 'date_next' to find consecutive days; load it with store.load_pairs(columns=...)")
    p = pairs.sort_values(KEY + ["date"], kind="stable").reset_index(drop=True)
    same = np.ones(len(p), dtype=bool)
    for k in KEY:
        v = p[k].to_numpy()
        same[1:] &= v[1:] == v[:-1]
    same[1:] &= p["date"].to_numpy()[1:] == p["date_next"].to_numpy()[:-1]
    same[0] = False
    p["episode"] = np.cumsum(~same) - 1
    return p


def _gross_and_notional(p: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Daily gross P&L, index notional traded, |change in hedge ratio| including set-up and unwind,
    and |change in hedge ratio| from day-to-day rebalancing only, for one delta column.
    ``p`` must come from ``sort_for_episodes``."""
    f = lambda c: p[c].to_numpy(dtype="float64")
    d, S, dS, dV = f(col), f("S"), f("dS"), f("dV")
    ep = p["episode"].to_numpy()
    first = np.r_[True, ep[1:] != ep[:-1]]
    last = np.r_[ep[1:] != ep[:-1], True]
    prev = np.r_[0.0, d[:-1]]
    prev[first] = 0.0                                   # set-up: buy delta_0 from nothing
    trade = np.abs(d - prev)
    rebalance = np.where(first, 0.0, trade)
    notional = trade * S
    notional[last] += np.abs(d[last]) * (S[last] + dS[last])   # unwind at the episode's final price
    turnover = trade.copy()
    turnover[last] += np.abs(d[last])
    return -dV + d * dS, notional, turnover, rebalance


def hedge_pnl(pairs: pd.DataFrame, delta_cols: list[str], cost_bp: float = 1.0,
              min_days: int = 5) -> pd.DataFrame:
    """Episode-level P&L of a short-option delta hedge for each delta column.

    Parameters
    ----------
    pairs : pairs table with ``date_next`` (or the output of ``sort_for_episodes``)
    delta_cols : e.g. ["delta", "delta_hw", "delta_nn_resid"]
    cost_bp : proportional cost in basis points of index notional traded
    min_days : drop episodes shorter than this many days
    """
    p = pairs if "episode" in pairs else sort_for_episodes(pairs)
    c = cost_bp * 1e-4
    g = p.groupby("episode", sort=True)
    out = g.agg(type=("type", "first"), expiry=("expiry", "first"), strike=("strike", "first"),
                start=("date", "first"), end=("date", "last"), n_days=("date", "size"))
    ep = np.unique(p["episode"].to_numpy(), return_inverse=True)[1]   # 0..k-1 even when p is a subset of episodes
    n = out["n_days"].to_numpy()
    for col in delta_cols:
        gross, notional, turnover, rebalance = _gross_and_notional(p, col)
        cost = c * notional
        pnl = gross - cost
        s1 = np.bincount(ep, weights=pnl)
        s2 = np.bincount(ep, weights=pnl**2)
        out[f"pnl_{col}"] = s1
        out[f"cost_{col}"] = np.bincount(ep, weights=cost)
        out[f"var_{col}"] = s2 / n - (s1 / n) ** 2
        out[f"turnover_{col}"] = np.bincount(ep, weights=turnover)
        out[f"rebal_{col}"] = np.bincount(ep, weights=rebalance)
    return out[out["n_days"] >= min_days].reset_index(drop=True)


def summarise(pnl: pd.DataFrame, delta_cols: list[str]) -> pd.DataFrame:
    """Per model: mean / std / 5% tail of episode P&L, mean cost, turnover and rebalancing per day."""
    rows = {}
    days = pnl["n_days"].sum()
    for col in delta_cols:
        x = pnl[f"pnl_{col}"]
        rows[col] = {
            "mean_pnl": x.mean(),
            "std_pnl": x.std(),
            "p05_pnl": x.quantile(0.05),
            "mean_cost": pnl[f"cost_{col}"].mean(),
            "turnover_per_day": pnl[f"turnover_{col}"].sum() / days,          # includes set-up and unwind
            "rebal_per_day": pnl[f"rebal_{col}"].sum() / (days - len(pnl)),   # day-to-day changes only
            "mean_daily_var": pnl[f"var_{col}"].mean(),
        }
    out = pd.DataFrame(rows).T
    out["n_episodes"] = len(pnl)
    return out


def cost_sweep(pairs: pd.DataFrame, delta_cols: list[str], bps=(0, 0.5, 1, 2, 5),
               min_days: int = 5) -> pd.DataFrame:
    """``summarise`` at several cost levels; index = (cost_bp, model)."""
    p = pairs if "episode" in pairs else sort_for_episodes(pairs)
    frames = []
    for bp in bps:
        s = summarise(hedge_pnl(p, delta_cols, bp, min_days), delta_cols)
        s["cost_bp"] = bp
        frames.append(s.reset_index(names="model"))
    return pd.concat(frames).set_index(["cost_bp", "model"])


def net_gain(pairs: pd.DataFrame, models: list[str], bps=(0, 0.5, 1, 2, 5),
             by: list | None = None) -> pd.DataFrame:
    """Gain ratio on net-of-cost daily P&L:  1 - sum(net_model^2) / sum(net_BS^2),  both at the same cost.

    At zero cost this is the gain ratio of ``evaluate.gain_table``.  A model that hedges better
    but trades more loses gain as the cost rises; the cost at which its gain reaches zero is its
    break-even.  ``by`` groups by columns of ``pairs`` (e.g. ["type"]).  Index = (cost_bp, *by)."""
    p = pairs if "episode" in pairs else sort_for_episodes(pairs)
    parts = {m: _gross_and_notional(p, _delta_col(m))[:2] for m in dict.fromkeys(["bs", *models])}
    keys = [p[b] for b in by] if by else None
    frames = []
    for bp in bps:
        se = pd.DataFrame({m: (g - bp * 1e-4 * n) ** 2 for m, (g, n) in parts.items()})
        if keys:
            tot = se.groupby(keys, observed=True).sum()
            t = pd.DataFrame({m: 1 - tot[m] / tot["bs"] for m in models})
            t.insert(0, "cost_bp", bp)
            frames.append(t.reset_index().set_index(["cost_bp", *by]))
        else:
            tot = se.sum()
            frames.append(pd.DataFrame([{m: 1 - tot[m] / tot["bs"] for m in models}],
                                       index=pd.Index([bp], name="cost_bp")))
    return pd.concat(frames)
