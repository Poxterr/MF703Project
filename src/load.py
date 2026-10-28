"""Stage 1: load, clean and standardise raw option-chain files.

The vendor files (``spx_eod_YYYYMM.txt``) have one row per (quote date, expiry,
strike) with the call and the put side by side.  This module:

1. parses the vendor format (bracketed headers, blank-string NAs, "1 x 365" size strings),
2. reshapes each row into two rows (one call, one put) with a common schema,
3. applies the sample filters used by Hull & White (2017) and Qiao & Wan (2024),
4. validates each file and prints a short quality report,
5. writes one Parquet file per input file.

Everything is parameterised so the same code runs on any date range, and a
different vendor layout only needs a new entry in ``COLUMN_MAPS``.

Command line::

    python -m src.load data/raw/*.txt --out data/processed/

Typical use from a notebook::

    from src.load import load_chain_files
    df = load_chain_files(sorted(glob("data/raw/spx_eod_2016*.txt")))
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Standard output schema                                                      #
# --------------------------------------------------------------------------- #
# Every loader, whatever the vendor, produces a frame with exactly these columns.
STANDARD_COLUMNS = [
    "date",        # quote date (datetime64)
    "expiry",      # expiry date (datetime64)
    "type",        # "C" or "P"
    "strike",      # float
    "S",           # underlying close
    "dte",         # calendar days to expiry (float, as the vendor reports it)
    "T",           # years to expiry = dte / 365
    "bid", "ask", "mid",
    "last", "volume",
    "bid_size", "ask_size",
    "iv",          # implied volatility (decimal)
    "delta", "gamma", "vega", "theta", "rho",
    "moneyness",   # S / K
]

# Vendor column names -> standard names, per option side.  Add a new vendor here.
COLUMN_MAPS = {
    "optionsdx": {
        "common": {
            "QUOTE_DATE": "date",
            "EXPIRE_DATE": "expiry",
            "STRIKE": "strike",
            "UNDERLYING_LAST": "S",
            "DTE": "dte",
        },
        "C": {
            "C_BID": "bid", "C_ASK": "ask", "C_LAST": "last", "C_VOLUME": "volume",
            "C_SIZE": "size", "C_IV": "iv", "C_DELTA": "delta", "C_GAMMA": "gamma",
            "C_VEGA": "vega", "C_THETA": "theta", "C_RHO": "rho",
        },
        "P": {
            "P_BID": "bid", "P_ASK": "ask", "P_LAST": "last", "P_VOLUME": "volume",
            "P_SIZE": "size", "P_IV": "iv", "P_DELTA": "delta", "P_GAMMA": "gamma",
            "P_VEGA": "vega", "P_THETA": "theta", "P_RHO": "rho",
        },
    },
}


# Unit conventions per vendor.  Checked on the 2016 files: vega is per 1% vol,
# theta is per calendar day, gamma is per index point.
VENDOR_UNITS = {
    "optionsdx": {"vega_scale": 100.0},
}


@dataclass
class FilterConfig:
    """Sample filters.  Defaults follow Hull & White (2017) / Qiao & Wan (2024)."""

    min_dte: float = 14.0          # drop options expiring within two weeks
    min_abs_delta: float = 0.05    # drop deep out-of-the-money
    max_abs_delta: float = 0.95    # drop deep in-the-money
    require_positive_bid: bool = True
    require_volume: bool = False   # True = traded-only subsample (robustness check)
    drop_crossed: bool = True      # bid > ask
    max_rel_spread: float | None = None  # e.g. 0.5 to drop spreads wider than 50% of mid


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #
def read_raw(path: str | Path) -> pd.DataFrame:
    """Read one vendor file as-is, with headers cleaned and blanks as NaN."""
    df = pd.read_csv(path, skipinitialspace=True, na_values=["", " "], low_memory=False)
    df.columns = [re.sub(r"[\[\]\s]", "", c) for c in df.columns]
    return df


def _split_size(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'12 x 365' -> (12, 365).  Missing or malformed -> NaN."""
    parts = s.astype("string").str.extract(r"(\d+)\s*x\s*(\d+)")
    return parts[0].astype(float), parts[1].astype(float)


def to_long(raw: pd.DataFrame, vendor: str = "optionsdx") -> pd.DataFrame:
    """Reshape a wide call/put file into the standard long schema (one row per option)."""
    cmap = COLUMN_MAPS[vendor]
    frames = []
    for side in ("C", "P"):
        cols = {**cmap["common"], **cmap[side]}
        part = raw[list(cols)].rename(columns=cols)
        part["type"] = side
        bid_size, ask_size = _split_size(part.pop("size"))
        part["bid_size"], part["ask_size"] = bid_size, ask_size
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    for c in ["strike", "S", "dte", "bid", "ask", "last", "volume",
              "iv", "delta", "gamma", "vega", "theta", "rho"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = df["volume"].fillna(0.0)
    # Vendor Greek units -> textbook units.  The vendor quotes vega per 1 percentage
    # point of volatility (ATM 30-day SPX vega ~ 2.4); Black-Scholes vega per unit of
    # volatility is 100x that (~ 230).  Hull-White's correction uses the latter.
    df["vega"] = df["vega"] * VENDOR_UNITS[vendor]["vega_scale"]
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["T"] = df["dte"] / 365.0
    df["moneyness"] = df["S"] / df["strike"]
    return df[STANDARD_COLUMNS]


# --------------------------------------------------------------------------- #
# Filtering and validation                                                    #
# --------------------------------------------------------------------------- #
def apply_filters(df: pd.DataFrame, cfg: FilterConfig = FilterConfig()) -> pd.DataFrame:
    """Apply the sample filters.  Returns a copy."""
    m = (df["dte"] >= cfg.min_dte) & df["delta"].abs().between(cfg.min_abs_delta, cfg.max_abs_delta)
    if cfg.require_positive_bid:
        m &= df["bid"] > 0
    if cfg.require_volume:
        m &= df["volume"] > 0
    if cfg.drop_crossed:
        m &= df["bid"] <= df["ask"]
    if cfg.max_rel_spread is not None:
        m &= (df["ask"] - df["bid"]) / df["mid"] <= cfg.max_rel_spread
    # After the delta filter no IV or Greek should be missing; drop any stragglers.
    m &= df[["iv", "delta", "vega", "gamma", "theta"]].notna().all(axis=1)
    return df.loc[m].reset_index(drop=True)


def validate(raw: pd.DataFrame, long: pd.DataFrame, filtered: pd.DataFrame, name: str = "") -> dict:
    """Compute a quality report for one file and print a one-line summary."""
    per_day = filtered.groupby("date").size()
    rep = {
        "file": name,
        "raw_rows": len(raw),
        "option_rows": len(long),
        "filtered_rows": len(filtered),
        "trading_days": int(long["date"].nunique()),
        "dup_keys": int(raw.duplicated(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"]).sum())
        if {"QUOTE_DATE", "EXPIRE_DATE", "STRIKE"} <= set(raw.columns) else np.nan,
        "crossed_quotes": int((long["bid"] > long["ask"]).sum()),
        "zero_bid_share": float((long["bid"] == 0).mean()),
        "iv_missing_after_filter": float(filtered["iv"].isna().mean()),
        "min_rows_per_day": int(per_day.min()) if len(per_day) else 0,
        "median_rows_per_day": int(per_day.median()) if len(per_day) else 0,
        "underlying_consistent": bool((long.groupby("date")["S"].nunique() == 1).all()),
    }
    print(f"[{name}] days={rep['trading_days']:>3}  raw={rep['raw_rows']:>7}  "
          f"filtered={rep['filtered_rows']:>7}  dup={rep['dup_keys']}  "
          f"crossed={rep['crossed_quotes']}  rows/day~{rep['median_rows_per_day']}")
    return rep


# --------------------------------------------------------------------------- #
# Public entry points                                                         #
# --------------------------------------------------------------------------- #
def load_chain_file(path: str | Path, cfg: FilterConfig = FilterConfig(),
                    vendor: str = "optionsdx", report: list | None = None) -> pd.DataFrame:
    """Parse, reshape, filter and validate one file."""
    raw = read_raw(path)
    long = to_long(raw, vendor)
    filtered = apply_filters(long, cfg)
    rep = validate(raw, long, filtered, Path(path).name)
    if report is not None:
        report.append(rep)
    return filtered


def load_chain_files(paths: Iterable[str | Path], cfg: FilterConfig = FilterConfig(),
                     vendor: str = "optionsdx", out_dir: str | Path | None = None) -> pd.DataFrame:
    """Load many files; optionally write each to Parquet in ``out_dir``.  Returns the stacked frame."""
    report: list[dict] = []
    frames = []
    for p in paths:
        df = load_chain_file(p, cfg, vendor, report)
        if out_dir is not None:
            out = Path(out_dir) / (Path(p).stem + ".parquet")
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "expiry", "strike", "type"])
    if out_dir is not None:
        pd.DataFrame(report).to_csv(Path(out_dir) / "load_report.csv", index=False)
    return df.reset_index(drop=True)


def find_raw_files(raw_dir: str | Path, years: Iterable[int] | None = None,
                   pattern: str = "spx_eod_*.txt") -> list[Path]:
    """All raw files under ``raw_dir``, searched recursively (so ``data/raw/2016/...`` etc.).

    ``years`` restricts to files whose name contains one of those years (``spx_eod_2017xx.txt``).
    """
    files = sorted(Path(raw_dir).rglob(pattern))
    if years is not None:
        ys = {str(y) for y in years}
        files = [f for f in files if any(f"_{y}" in f.name for y in ys)]
    return files


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="raw vendor files (omit to search --raw-dir)")
    ap.add_argument("--raw-dir", default="data/raw", help="searched recursively when no files are given")
    ap.add_argument("--years", nargs="*", type=int, help="e.g. --years 2016 2017 (default: all)")
    ap.add_argument("--out", default="data/processed", help="output directory for Parquet")
    ap.add_argument("--vendor", default="optionsdx", choices=list(COLUMN_MAPS))
    ap.add_argument("--min-dte", type=float, default=14.0)
    ap.add_argument("--min-delta", type=float, default=0.05)
    ap.add_argument("--max-delta", type=float, default=0.95)
    ap.add_argument("--traded-only", action="store_true", help="keep only quotes with volume > 0")
    a = ap.parse_args(argv)
    cfg = FilterConfig(min_dte=a.min_dte, min_abs_delta=a.min_delta,
                       max_abs_delta=a.max_delta, require_volume=a.traded_only)
    files = sorted(a.files) if a.files else find_raw_files(a.raw_dir, a.years)
    if not files:
        raise SystemExit(f"no raw files found under {a.raw_dir} (years={a.years})")
    df = load_chain_files(files, cfg, a.vendor, a.out)
    print(f"\nTotal filtered rows: {len(df):,}  "
          f"({df['date'].min().date()} to {df['date'].max().date()})")


if __name__ == "__main__":
    main()
