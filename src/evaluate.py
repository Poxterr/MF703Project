"""Stage 4: evaluation harness.

The headline metric is the Hull & White (2017) gain ratio

    G = 1 - SSE(model) / SSE(Black-Scholes),   SSE = sum (dV - delta * dS)^2

computed on any subset (year, split, call/put, delta bucket, maturity bucket).
Every model's predictions are stored as a column ``delta_<name>`` in the pairs
table, so this module only ever needs the table and a list of model names.

Nothing here copies the pairs table: the functions build a small frame of squared errors
(one column per model) and group that, so they are safe on the full multi-million-row table.

Two functions put the gain ratios in context:

``vega_attribution``   how much of the Black-Scholes hedging error the realised implied-vol
                       change accounts for (uses information from t+1: a diagnostic, not a hedge)
``in_sample_ceiling``  the gain a model class reaches when it is fitted on the very period it is
                       scored on: an upper bound for what that class can do out of sample
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Columns the evaluation needs (see src/store.py for the full modelling set).
EVAL_COLUMNS = ["type", "date", "year", "split", "dte", "delta", "dV", "dS", "delta_bucket", "ttm_class"]

CRISIS_START, CRISIS_END = "2020-02-19", "2020-04-30"   # S&P 500 peak to the end of April 2020


def add_predictions(pairs: pd.DataFrame, models: dict, inplace: bool = False) -> pd.DataFrame:
    """Write ``delta_<name>`` columns for each fitted model in ``models``.

    ``inplace=True`` avoids copying a multi-million-row table."""
    if not inplace:
        pairs = pairs.copy()
    for name, m in models.items():
        pairs[f"delta_{name}"] = np.asarray(m.predict(pairs), dtype="float32")
    return pairs


def _delta_col(model: str) -> str:
    return "delta" if model == "bs" else f"delta_{model}"


def squared_errors(pairs: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """One float64 column of squared one-step hedging errors per model, plus 'bs'."""
    dV = pairs["dV"].to_numpy(dtype="float64")
    dS = pairs["dS"].to_numpy(dtype="float64")
    out = {}
    for m in dict.fromkeys(["bs", *models]):
        out[m] = (dV - pairs[_delta_col(m)].to_numpy(dtype="float64") * dS) ** 2
    return pd.DataFrame(out, index=pairs.index)


def sse(pairs: pd.DataFrame, col: str) -> float:
    return float(((pairs["dV"].astype("float64") - pairs[col].astype("float64") * pairs["dS"].astype("float64")) ** 2).sum())


def gain_ratio(pairs: pd.DataFrame, model: str, benchmark: str = "delta") -> float:
    return 1.0 - sse(pairs, _delta_col(model)) / sse(pairs, benchmark)


def _group_keys(pairs: pd.DataFrame, by: list) -> list:
    """Column names -> the columns themselves; Series are passed through (lets callers group by a
    derived key such as |delta bucket| without adding a column to the big table)."""
    return [pairs[b] if isinstance(b, str) else b for b in by]


def gain_table(pairs: pd.DataFrame, models: list[str], by: list | None = None) -> pd.DataFrame:
    """Gain ratios for each model, overall or grouped.

    ``by`` may mix column names and Series aligned with ``pairs`` (name the Series)."""
    se = squared_errors(pairs, models)
    if not by:
        tot = se.sum()
        row = {m: 1.0 - tot[m] / tot["bs"] for m in models}
        return pd.DataFrame([row | {"n": len(se)}], index=["overall"])
    g = se.groupby(_group_keys(pairs, by), observed=True)
    tot, n = g.sum(), g.size()
    out = pd.DataFrame({m: 1.0 - tot[m] / tot["bs"] for m in models})
    out["n"] = n
    return out


def abs_delta_bucket(pairs: pd.DataFrame) -> pd.Series:
    """Unsigned delta bucket (0.1 ... 0.9) as a Series, for plots that put calls and puts on one axis."""
    if "delta_bucket" in pairs:
        b = pairs["delta_bucket"].abs()
    else:
        from .build_dataset import delta_bucket
        b = delta_bucket(pairs["delta"]).abs()
    return b.round(1).rename("abs_delta_bucket")


def with_buckets(pairs: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``pairs`` with ``abs_delta_bucket`` (and the pipeline's bucket columns if missing).
    Convenient for small frames; on the full table use ``abs_delta_bucket(pairs)`` as a group key."""
    from .build_dataset import delta_bucket, ttm_class
    pairs = pairs.copy()
    if "delta_bucket" not in pairs:
        pairs["delta_bucket"] = delta_bucket(pairs["delta"])
    if "ttm_class" not in pairs:
        pairs["ttm_class"] = ttm_class(pairs["dte"])
    pairs["abs_delta_bucket"] = abs_delta_bucket(pairs)
    return pairs


def crisis_mask(pairs: pd.DataFrame, start: str = CRISIS_START, end: str = CRISIS_END) -> pd.Series:
    return (pairs["date"] >= start) & (pairs["date"] <= end)


def crisis_window(pairs: pd.DataFrame, start: str = CRISIS_START, end: str = CRISIS_END) -> pd.DataFrame:
    return pairs[crisis_mask(pairs, start, end)]


def standard_report(pairs: pd.DataFrame, models: list[str], out_dir: str | Path | None = None) -> dict:
    """The tables every modelling notebook produces.  Returns a dict; writes CSVs if ``out_dir``.

    by_split_type   train / val / test x call / put          (the headline table)
    by_year_type    each calendar year x call / put          (2020 dominates pooled numbers)
    by_delta_bucket call / put x |delta| bucket, test split only
    by_maturity     call / put x maturity class, test split only
    crisis          call / put inside the Feb-Apr 2020 window
    """
    year = pairs["year"] if "year" in pairs else pairs["date"].dt.year.rename("year")
    test = pairs[pairs["split"] == "test"] if "split" in pairs else pairs
    if "ttm_class" in test:
        ttm = test["ttm_class"]
    else:
        from .build_dataset import ttm_class
        ttm = ttm_class(test["dte"]).rename("ttm_class")
    tables = {"by_year_type": gain_table(pairs, models, [year, "type"])}
    if "split" in pairs:
        tables = {"by_split_type": gain_table(pairs, models, ["split", "type"]), **tables}
    tables["by_delta_bucket"] = gain_table(test, models, ["type", abs_delta_bucket(test)])
    tables["by_maturity"] = gain_table(test, models, ["type", ttm])
    crisis = crisis_window(pairs)
    if len(crisis):
        tables["crisis"] = gain_table(crisis, models, ["type"])
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, t in tables.items():
            t.to_csv(out_dir / f"gain_{k}.csv")
    return tables


# --------------------------------------------------------------------------- #
# Putting the gain ratios in context                                          #
# --------------------------------------------------------------------------- #
def vega_attribution(pairs: pd.DataFrame, by: list | None = None) -> pd.DataFrame:
    """Share of the Black-Scholes hedging error explained by the realised implied-vol change.

    First-order Black-Scholes attribution of the option's one-step price change:

        dV ~= delta * dS + vega * d_iv + (gamma, theta, noise)

    ``vega_share`` = 1 - SSE(dV - delta*dS - vega*d_iv) / SSE(dV - delta*dS).

    This uses d_iv, the IV change over the hedging interval, which is not known at time t, so it
    is not an achievable gain.  It says how much of the error is about volatility at all: high in
    2020 (about 0.6), low in calm years.

    Read it as an indicator, not a strict bound.  The vendor's implied vols carry their own
    day-to-day noise, and vega * (noisy d_iv) adds error; in a quiet year that noise can outweigh
    the signal and the share comes out near zero or negative, even though a smooth correction such
    as Hull-White still earns a positive gain.
    """
    f = lambda c: pairs[c].to_numpy(dtype="float64")
    e_bs = f("dV") - f("delta") * f("dS")
    e_dv = e_bs - f("vega") * f("d_iv")
    se = pd.DataFrame({"bs": e_bs**2, "delta_vega": e_dv**2}, index=pairs.index)
    if not by:
        tot = se.sum()
        return pd.DataFrame([{"vega_share": 1 - tot["delta_vega"] / tot["bs"], "n": len(se)}], index=["overall"])
    g = se.groupby(_group_keys(pairs, by), observed=True)
    tot = g.sum()
    return pd.DataFrame({"vega_share": 1 - tot["delta_vega"] / tot["bs"], "n": g.size()})


def in_sample_ceiling(pairs: pd.DataFrame, make_model: Callable[[], object], name: str,
                      period: str = "year") -> pd.DataFrame:
    """Fit a fresh model on each period and score it on that same period.

    No out-of-sample model of the same class can be expected to beat this, so the gap between a
    model's out-of-sample gain and its ceiling measures what is lost to the relationship moving
    over time (regime change), as opposed to what the model class cannot represent at all.

    ``make_model`` returns an unfitted model, e.g. ``models.HullWhite``.
    Returns gain by (period, type) in a column ``<name>_ceiling``.
    """
    key = pairs[period] if period in pairs else pairs["date"].dt.year
    rows = []
    for val, p in pairs.groupby(key, observed=True):
        m = make_model().fit(p)
        pred = np.asarray(m.predict(p), dtype="float64")
        dV, dS = p["dV"].to_numpy(dtype="float64"), p["dS"].to_numpy(dtype="float64")
        se = pd.DataFrame({"bs": (dV - p["delta"].to_numpy(dtype="float64") * dS) ** 2,
                           "m": (dV - pred * dS) ** 2, "type": p["type"].to_numpy()})
        tot = se.groupby("type", observed=True)[["bs", "m"]].sum()
        for t, r in tot.iterrows():
            rows.append({period: val, "type": t, f"{name}_ceiling": 1 - r["m"] / r["bs"]})
    return pd.DataFrame(rows).set_index([period, "type"])


def plot_gain_by_bucket(table: pd.DataFrame, models: list[str], title: str, ax=None):
    """Line plot of gain ratio vs delta bucket, one line per model.
    Expects a table indexed by bucket (last index level), e.g. ``gain_table(p, models, [abs_delta_bucket(p)])``."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    for m in models:
        ax.plot(table.index.get_level_values(-1), table[m], marker="o", label=m)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("|delta_BS| bucket"); ax.set_ylabel("gain ratio"); ax.set_title(title); ax.legend()
    return ax
