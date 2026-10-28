"""Stage 5b: multi-day hedging simulation with transaction costs.

The one-day loss used for training has no rebalancing in it.  To judge a hedge
net of costs we follow each contract for its whole life in the sample:

    day t:   hold -1 option, +delta_t units of index   (short option, long hedge)
    P&L_t  = -(V_{t+1} - V_t) + delta_t (S_{t+1} - S_t) - c * |delta_t - delta_{t-1}| * S_t

with proportional cost ``c`` per unit of index notional traded (e.g. 1 bp = 1e-4).
The first day pays c * |delta_0| S_0 to set up the hedge.

Output per (contract, model): total P&L, total cost, realised variance of daily P&L,
so you can plot "net P&L distribution" and "cost vs. error reduction" across models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pairs import KEY


def hedge_pnl(pairs: pd.DataFrame, delta_cols: list[str], cost_bp: float = 1.0,
              min_days: int = 5) -> pd.DataFrame:
    """Path-wise P&L of a short-option delta hedge for each model's delta column.

    Parameters
    ----------
    pairs : pairs table (must be consecutive-day pairs of the same contract)
    delta_cols : e.g. ["delta", "delta_hw", "delta_nn_resid"]
    cost_bp : proportional cost in basis points of index notional traded
    min_days : ignore contracts observed fewer days than this
    """
    c = cost_bp * 1e-4
    out = []
    for key, g in pairs.sort_values("date").groupby(KEY, sort=False):
        if len(g) < min_days:
            continue
        dV, dS, S = g["dV"].to_numpy(), g["dS"].to_numpy(), g["S"].to_numpy()
        row = {"type": key[0], "expiry": key[1], "strike": key[2], "n_days": len(g),
               "start": g["date"].iloc[0], "end": g["date"].iloc[-1]}
        for col in delta_cols:
            d = g[col].to_numpy()
            trade = np.abs(np.diff(d, prepend=0.0))         # includes initial set-up
            cost = c * trade * S
            pnl = -dV + d * dS - cost
            row[f"pnl_{col}"] = pnl.sum()
            row[f"cost_{col}"] = cost.sum()
            row[f"var_{col}"] = pnl.var()
            row[f"turnover_{col}"] = trade.sum()
        out.append(row)
    return pd.DataFrame(out)


def summarise(pnl: pd.DataFrame, delta_cols: list[str]) -> pd.DataFrame:
    """Mean / std of total P&L, mean cost and turnover per model."""
    rows = {}
    for col in delta_cols:
        rows[col] = {
            "mean_pnl": pnl[f"pnl_{col}"].mean(),
            "std_pnl": pnl[f"pnl_{col}"].std(),
            "mean_cost": pnl[f"cost_{col}"].mean(),
            "mean_turnover": pnl[f"turnover_{col}"].mean(),
            "mean_daily_var": pnl[f"var_{col}"].mean(),
        }
    return pd.DataFrame(rows).T


def cost_sweep(pairs: pd.DataFrame, delta_cols: list[str], bps=(0, 0.5, 1, 2, 5)) -> pd.DataFrame:
    """Summary table for several cost levels; index = (cost_bp, model)."""
    frames = []
    for bp in bps:
        s = summarise(hedge_pnl(pairs, delta_cols, bp), delta_cols)
        s["cost_bp"] = bp
        frames.append(s.reset_index(names="model"))
    return pd.concat(frames).set_index(["cost_bp", "model"])
