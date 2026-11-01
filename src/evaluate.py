"""Stage 4: evaluation harness.

The headline metric is the Hull & White (2017) gain ratio

    G = 1 - SSE(model) / SSE(Black-Scholes),   SSE = sum (dV - delta * dS)^2

computed on any subset (year, split, call/put, delta bucket, maturity bucket).
Every model's predictions are stored as a column ``delta_<name>`` in the pairs
table, so this module only ever needs the table and a list of model names.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Bucket columns (``delta_bucket``, signed, and ``ttm_class``) are produced by the pipeline
# (src/build_dataset.py) so every notebook uses the same definition.  ``abs_delta_bucket`` below
# is the unsigned version for plots that pool calls and puts.


# Columns the evaluation needs; read only these from pairs.parquet to keep memory small:
#   pd.read_parquet(path, columns=evaluate.EVAL_COLUMNS + model feature columns)
EVAL_COLUMNS = ["type", "date", "year", "split", "dte", "delta", "dV", "dS", "delta_bucket", "ttm_class"]


def add_predictions(pairs: pd.DataFrame, models: dict, inplace: bool = False) -> pd.DataFrame:
    """Write ``delta_<name>`` columns for each fitted model in ``models``.

    ``inplace=True`` avoids copying a multi-million-row table."""
    if not inplace:
        pairs = pairs.copy()
    for name, m in models.items():
        pairs[f"delta_{name}"] = m.predict(pairs).astype("float32")
    return pairs


def sse(pairs: pd.DataFrame, col: str) -> float:
    return float(((pairs["dV"] - pairs[col] * pairs["dS"]) ** 2).sum())


def gain_ratio(pairs: pd.DataFrame, model: str, benchmark: str = "delta") -> float:
    return 1.0 - sse(pairs, f"delta_{model}" if model != "bs" else "delta") / sse(pairs, benchmark)


def gain_table(pairs: pd.DataFrame, models: list[str], by: list[str] | None = None) -> pd.DataFrame:
    """Gain ratios for each model, overall or grouped by the columns in ``by``."""
    def _one(p):
        return pd.Series({m: gain_ratio(p, m) for m in models} | {"n": len(p)})
    if not by:
        return _one(pairs).to_frame("overall").T
    cols = [c for c in {"dV", "dS", "delta", *[f"delta_{m}" for m in models if m != "bs"]} if c in pairs]
    return pairs.groupby(by, observed=True)[cols].apply(_one)


def with_buckets(pairs: pd.DataFrame) -> pd.DataFrame:
    """Ensure the pipeline's bucket columns exist (recompute only if a frame lacks them) and add
    ``abs_delta_bucket``."""
    from .build_dataset import delta_bucket, ttm_class
    pairs = pairs.copy()
    if "delta_bucket" not in pairs:
        pairs["delta_bucket"] = delta_bucket(pairs["delta"])
    if "ttm_class" not in pairs:
        pairs["ttm_class"] = ttm_class(pairs["dte"])
    pairs["abs_delta_bucket"] = pairs["delta_bucket"].abs()
    return pairs


def standard_report(pairs: pd.DataFrame, models: list[str], out_dir: str | Path | None = None) -> dict:
    """The tables every notebook should produce.  Returns them in a dict; writes CSVs if ``out_dir``."""
    p = with_buckets(pairs)
    tables = {
        "by_split_type": gain_table(p, models, ["split", "type"]),
        "by_year_type": gain_table(p.assign(year=p["date"].dt.year), models, ["year", "type"]),
        "by_delta_bucket": gain_table(p, models, ["type", "abs_delta_bucket"]),
        "by_maturity": gain_table(p, models, ["type", "ttm_class"]),
    }
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        for k, t in tables.items():
            t.to_csv(out_dir / f"gain_{k}.csv")
    return tables


def crisis_window(pairs: pd.DataFrame, start="2020-02-19", end="2020-04-30") -> pd.DataFrame:
    return pairs[(pairs["date"] >= start) & (pairs["date"] <= end)]


def plot_gain_by_bucket(table: pd.DataFrame, models: list[str], title: str, ax=None):
    """Line plot of gain ratio vs delta bucket, one line per model.  Expects ``gain_table(..., ['abs_delta_bucket'])``."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    for m in models:
        ax.plot(table.index.get_level_values(-1), table[m], marker="o", label=m)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("|delta_BS| bucket"); ax.set_ylabel("gain ratio"); ax.set_title(title); ax.legend()
    return ax
