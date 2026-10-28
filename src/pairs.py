"""Stage 2: pair each option with its next-trading-day quote and build the hedging target.

For option *i* quoted on day *t* and again on the next trading day *t+1*:

    dV = V_{t+1} - V_t          change in the option's mid price
    dS = S_{t+1} - S_t          change in the underlying
    y  = dV / dS                the "realised" hedge ratio for this one-day move

A hedge ratio delta has one-day hedging error  eps = dV - delta * dS.
Minimising the mean squared error is a weighted least-squares problem in y:

    (dV - delta * dS)^2 = dS^2 * (y - delta)^2

so every model in ``models.py`` fits y (or y - delta_BS for residual models)
with sample weights dS^2.

The function also stamps a chronological ``split`` column (train / val / test)
so no notebook can choose its own split by accident.

Command line::

    python -m src.pairs data/processed/ --out data/processed/pairs.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["type", "expiry", "strike"]
FEATURES = ["T", "delta", "moneyness", "iv", "theta", "vega", "gamma"]   # as in Qiao & Wan (2024)

DEFAULT_SPLITS = {          # inclusive year ranges
    "train": (2016, 2018),
    "val": (2019, 2019),
    "test": (2020, 2020),
}


def next_trading_day_map(dates: pd.Series) -> dict:
    """Map each trading date in the data to the next one present in the data."""
    d = np.sort(dates.unique())
    return dict(zip(d[:-1], d[1:]))


def build_pairs(df: pd.DataFrame, min_abs_ds: float = 0.0) -> pd.DataFrame:
    """Join each (type, expiry, strike) quote to the same contract on the next trading day.

    Parameters
    ----------
    df : filtered long frame from ``src.load``
    min_abs_ds : drop days where the index moved less than this (0 keeps everything;
                 the weighting dS^2 already makes tiny moves nearly irrelevant)
    """
    nxt = next_trading_day_map(df["date"])
    left = df.copy()
    left["date_next"] = left["date"].map(nxt)
    left = left.dropna(subset=["date_next"])

    right = df[KEY + ["date", "mid", "S"]].rename(
        columns={"date": "date_next", "mid": "mid_next", "S": "S_next"})
    pairs = left.merge(right, on=KEY + ["date_next"], how="inner")

    pairs["dV"] = pairs["mid_next"] - pairs["mid"]
    pairs["dS"] = pairs["S_next"] - pairs["S"]
    pairs = pairs[pairs["dS"].abs() > min_abs_ds]
    pairs["y"] = pairs["dV"] / pairs["dS"]            # realised hedge ratio
    pairs["w"] = pairs["dS"] ** 2                      # WLS weight
    pairs["err_bs"] = pairs["dV"] - pairs["delta"] * pairs["dS"]   # benchmark error
    return pairs.reset_index(drop=True)


def assign_splits(pairs: pd.DataFrame, splits: dict = DEFAULT_SPLITS) -> pd.DataFrame:
    """Add a ``split`` column from year ranges.  Rows outside every range get 'unused'."""
    year = pairs["date"].dt.year
    split = pd.Series("unused", index=pairs.index)
    for name, (y0, y1) in splits.items():
        split[(year >= y0) & (year <= y1)] = name
    pairs = pairs.copy()
    pairs["split"] = split
    return pairs


def make_pairs_table(processed_dir: str | Path, out: str | Path | None = None,
                     splits: dict = DEFAULT_SPLITS) -> pd.DataFrame:
    """Read every Parquet in ``processed_dir`` (except pairs.parquet), build and save the pairs table."""
    files = sorted(p for p in Path(processed_dir).glob("*.parquet") if p.name != "pairs.parquet")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    pairs = assign_splits(build_pairs(df), splits)
    if out is not None:
        pairs.to_parquet(out, index=False)
    print(f"pairs: {len(pairs):,} rows, "
          f"{pairs['date'].min().date()} to {pairs['date'].max().date()}")
    print(pairs.groupby(["split", "type"]).size().unstack(fill_value=0))
    return pairs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("processed_dir")
    ap.add_argument("--out", default="data/processed/pairs.parquet")
    a = ap.parse_args(argv)
    make_pairs_table(a.processed_dir, a.out)


if __name__ == "__main__":
    main()
