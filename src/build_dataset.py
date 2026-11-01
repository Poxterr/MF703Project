"""Build the modelling dataset from the raw optionsDX files, end to end ("mother script").

Run from the repo root, for example::

    python -m src.build_dataset                       # all years found under data/raw, 1-day horizon
    python -m src.build_dataset --years 2019 2020     # a subset of years
    python -m src.build_dataset --horizon 5           # weekly hedging pairs (5 trading days)
    python -m src.build_dataset --splits paper        # train 2010-17 / val 2018 / test 2019
    python -m src.build_dataset --stage pairs         # skip stage 1 if Parquet files already exist

What it produces (all under data/processed/):

    spx_eod_YYYYMM.parquet      stage 1: one cleaned file per raw file (one row per option-day)
    load_report.csv             stage 1: one quality row per raw file
    data_quality.md             stage 1: human-readable list of every day that was missing or dropped
    pairs_h1_parts/year=YYYY/   stage 2: pairs for one year (handy for loading a single year)
    pairs.parquet               stage 2: all pairs in one file - the table every model trains on
    pairs_summary.csv           stage 2: counts and benchmark error by year

How the pipeline is organised
-----------------------------
Stage 1 - CLEAN (one raw file at a time, so memory stays small)
    read_raw   : vendor text -> DataFrame                         (src.load)
    to_long    : one row per strike -> one row per option (C / P) (src.load)
    calendar   : drop rows dated on days the exchange was closed  (new, this file)
    filters    : Hull-White / Qiao-Wan sample filters             (src.load)
    report     : counts, missing days, zero-bid days              (extended here)

Stage 2 - PAIR (one year at a time)
    For option i quoted on trading day t, find the SAME contract (type, expiry, strike)
    exactly h trading days later, using the exchange calendar - never "the next date
    that happens to be in the file".  Then

        dV = mid_{t+h} - mid_t        change in option price (index points)
        dS = S_{t+h}   - S_t          change in the index
        y  = dV / dS                  realised hedge ratio for this move
        w  = dS^2                     weight, so that  w * (y - delta)^2 = (dV - delta*dS)^2

    which is exactly the one-step hedging error in Qiao & Wan (2024) eq. (1).  Every
    model fits y (or y - delta_BS for residual models) with weights w.

Stage 2 also adds the analysis columns the paper reports on (delta bucket, maturity class),
the sentiment features of Chen & Li (2023) when available (index log-return; VIX if a FRED
CSV is supplied), and a chronological train / val / test split so no notebook can pick its
own split by accident.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import load as L   # the team's parsing / filtering code - reused, not copied

# --------------------------------------------------------------------------- #
# 0. Configuration                                                            #
# --------------------------------------------------------------------------- #
KEY = ["type", "expiry", "strike"]            # what identifies one contract in optionsDX files
FEATURES = ["T", "delta", "moneyness", "iv", "theta", "vega", "gamma"]   # Qiao & Wan, Table 2
SENTIMENT = ["index_ret", "vix"]              # Chen & Li (2023) "Fea3-CL" extras

# Named split sets (inclusive year ranges).  Chronological on purpose: the same contract
# appears on consecutive days, so a random split would leak tomorrow into today.
SPLIT_SETS = {
    "paper":  {"train": (2010, 2017), "val": (2018, 2018), "test": (2019, 2019)},   # mirrors the paper's window
    "modern": {"train": (2010, 2019), "val": (2020, 2020), "test": (2021, 2023)},   # the regimes the paper never saw
    "readme": {"train": (2016, 2018), "val": (2019, 2019), "test": (2020, 2020)},   # the repo's original split
}


@dataclass
class BuildConfig:
    raw_dir: Path = Path("data/raw")
    out_dir: Path = Path("data/processed")
    years: list[int] | None = None            # None = every year found under raw_dir
    horizon: int = 1                          # hedging interval in TRADING days (1 = daily, 5 ~ weekly, 21 ~ monthly)
    splits: str = "modern"
    filters: L.FilterConfig = field(default_factory=L.FilterConfig)
    vix_csv: Path | None = None               # optional FRED VIXCLS.csv for the Chen-Li call feature
    min_abs_ds: float = 0.0                   # drop pairs where the index did not move at all


# --------------------------------------------------------------------------- #
# 1. The exchange calendar                                                    #
# --------------------------------------------------------------------------- #
def trading_days(start: str = "2009-12-01", end: str = "2024-12-31") -> pd.DatetimeIndex:
    """All NYSE trading days in [start, end].

    Why we need this: the vendor files are missing some real trading days and contain
    rows on some days the market was closed (holidays, Hurricane Sandy 2012, 5 Dec 2018).
    Pairing "day t with the next day in the file" would then produce two-day moves labelled
    as one-day hedges, or pairs across a day on which nothing traded.
    """
    try:
        import pandas_market_calendars as pmc
        sched = pmc.get_calendar("NYSE").schedule(start, end)
        return pd.DatetimeIndex(sched.index).normalize()
    except ImportError:
        # Fallback (approximate): US federal holidays are not the NYSE calendar
        # (NYSE closes on Good Friday, stays open on Columbus and Veterans Day).
        print("WARNING: pandas_market_calendars not installed; using an approximate calendar. "
              "pip install pandas_market_calendars")
        from pandas.tseries.holiday import USFederalHolidayCalendar
        hol = USFederalHolidayCalendar().holidays(start, end)
        return pd.bdate_range(start, end).difference(hol)


def shift_trading_days(dates: pd.Series, cal: pd.DatetimeIndex, h: int) -> pd.Series:
    """Map each date to the trading day exactly ``h`` positions later (or earlier if h < 0) on ``cal``.

    Dates that are not on the calendar map to NaT.  This is the heart of the fix: the
    target date is computed from the calendar, so a vendor gap can only *drop* a pair,
    never stretch it.
    """
    pos = pd.Series(np.arange(len(cal)), index=cal)        # trading day -> position 0,1,2,...
    idx = dates.map(pos)                                   # NaN for non-trading dates
    tgt = idx + h
    ok = tgt.notna() & (tgt >= 0) & (tgt < len(cal))
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    out[ok] = cal[tgt[ok].astype(int).to_numpy()]
    return out


# --------------------------------------------------------------------------- #
# 2. Stage 1 - clean one raw file                                             #
# --------------------------------------------------------------------------- #
def clean_file(path: Path, cfg: BuildConfig, cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Parse, reshape, calendar-check, filter and validate one monthly file.

    Returns the filtered frame and a quality-report row.  Every step is one line so the
    order of operations is visible: parse -> long -> calendar -> filters.
    """
    raw = L.read_raw(path)                                 # vendor text, bracketed headers removed
    long = L.to_long(raw)                                  # one row per option, standard column names

    # (a) rows dated on days the exchange was closed -> drop, but remember them
    on_cal = long["date"].isin(cal)
    non_trading_dates = sorted(long.loc[~on_cal, "date"].dt.date.unique())
    long = long[on_cal]

    # (b) days where the vendor shows bid = 0 on every option (Jan-Feb 2010): the bid > 0
    #     filter removes the whole day, so flag it rather than let it vanish silently
    bid_share = long.groupby("date")["bid"].apply(lambda s: (s > 0).mean())
    zero_bid_days = sorted(bid_share[bid_share == 0].index.date)

    # (c) trading days the file should contain (by its year-month) but does not
    ym = path.stem[-6:]                                    # 'spx_eod_201903' -> '201903'
    expected = cal[(cal.year == int(ym[:4])) & (cal.month == int(ym[4:]))]
    missing_days = sorted(set(expected.date) - set(long["date"].dt.date.unique()))

    # (d) the sample filters (dte >= 14, 0.05 <= |delta| <= 0.95, bid > 0, ...)
    filtered = L.apply_filters(long, cfg.filters)

    rep = L.validate(raw, long, filtered, path.name)        # the team's standard report row ...
    rep.update({                                           # ... plus the calendar findings
        "non_trading_rows_dropped": int((~on_cal).sum()),
        "non_trading_dates": ";".join(map(str, non_trading_dates)),
        "missing_trading_days": ";".join(map(str, missing_days)),
        "zero_bid_days": ";".join(map(str, zero_bid_days)),
    })
    return filtered.reset_index(drop=True), rep


def run_stage1(cfg: BuildConfig, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Clean every raw file, write one Parquet per file, the report CSV and data_quality.md."""
    files = L.find_raw_files(cfg.raw_dir, cfg.years)
    if not files:
        raise SystemExit(f"no raw files under {cfg.raw_dir} for years={cfg.years}")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for p in files:
        df, rep = clean_file(p, cfg, cal)
        df.to_parquet(cfg.out_dir / f"{p.stem}.parquet", index=False)
        rows.append(rep)
    report = pd.DataFrame(rows)
    # merge with any existing report so partial runs (--years) do not wipe other years
    rpt_path = cfg.out_dir / "load_report.csv"
    if rpt_path.exists():
        old = pd.read_csv(rpt_path)
        report = pd.concat([old[~old["file"].isin(report["file"])], report], ignore_index=True)
    report = report.sort_values("file").reset_index(drop=True)
    report.to_csv(rpt_path, index=False)
    write_quality_md(report, cfg.out_dir / "data_quality.md")
    print(f"stage 1: {len(files)} files cleaned in {time.time() - t0:.0f}s")
    return report


def write_quality_md(report: pd.DataFrame, path: Path) -> None:
    """A short markdown note listing every day that was missing, dropped or empty."""
    r = report.copy()
    r["year"] = r["file"].str[8:12]
    lines = ["# Data quality (generated by src/build_dataset.py)", "",
             "| year | files | trading days in data | filtered rows | crossed quotes | non-trading rows dropped | missing trading days |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for y, g in r.groupby("year"):
        n_missing = sum(len(s.split(";")) for s in g["missing_trading_days"].fillna("") if s)
        lines.append(f"| {y} | {len(g)} | {int(g['trading_days'].sum())} | {int(g['filtered_rows'].sum()):,} | "
                     f"{int(g['crossed_quotes'].sum())} | {int(g['non_trading_rows_dropped'].sum())} | {n_missing} |")
    for col, title in [("missing_trading_days", "Trading days absent from the vendor files"),
                       ("zero_bid_days", "Days with bid = 0 on every option (removed by the bid > 0 filter)"),
                       ("non_trading_dates", "Rows on days the exchange was closed (dropped)")]:
        items = [d for s in r[col].fillna("") if s for d in s.split(";")]
        lines += ["", f"## {title} ({len(items)})", ""] + [f"- {d}" for d in sorted(set(items))]
    lines += ["", "Pairs are built on the NYSE calendar, so every day listed above simply has no pair "
              "(the day before it loses its forward pair); nothing is bridged across a gap."]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# 3. Stage 2 - pairs, features, splits                                        #
# --------------------------------------------------------------------------- #
def _read_year(out_dir: Path, year: int, months: range = range(1, 13)) -> pd.DataFrame:
    files = [out_dir / f"spx_eod_{year}{m:02d}.parquet" for m in months]
    files = [f for f in files if f.exists()]
    if not files:
        return pd.DataFrame(columns=L.STANDARD_COLUMNS)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def delta_bucket(delta: pd.Series) -> pd.Series:
    """Paper's Table 1 buckets: bucket d holds |delta| in [d - 0.05, d + 0.05), d = 0.1 ... 0.9.

    Returned with the sign of delta, so puts land in -0.1 ... -0.9 like the paper's tables.
    """
    b = np.floor((delta.abs() + 0.05) * 10 + 1e-9) / 10
    b = b.clip(0.1, 0.9)
    return np.sign(delta) * b


def ttm_class(dte: pd.Series) -> pd.Series:
    """Paper's Fig. 3 maturity classes from calendar days to expiry."""
    bins = [0, 30, 91, 182, 365, 730, np.inf]
    labels = ["0-1m", "1-3m", "3-6m", "6m-1y", "1-2y", ">2y"]
    return pd.cut(dte, bins=bins, labels=labels, right=False)


def daily_index(df: pd.DataFrame) -> pd.Series:
    """One underlying level per trading day (the loader already checks it is unique per day)."""
    return df.groupby("date")["S"].first().sort_index()


# Columns carried into the pairs table.  Quote-size fields etc. stay in the stage-1 Parquet files.
PAIR_COLUMNS = KEY + ["date", "S", "dte", "T", "bid", "ask", "mid", "volume",
                      "iv", "delta", "gamma", "vega", "theta", "rho", "moneyness"]
NEXT_COLUMNS = {"date": "date_next", "mid": "mid_next", "S": "S_next", "iv": "iv_next", "delta": "delta_next"}


def _pair_one_side(left: pd.DataFrame, nxt: pd.DataFrame, cal: pd.DatetimeIndex, cfg: BuildConfig) -> pd.DataFrame:
    """Join each quote to the same contract h trading days later, one month of start dates at a time.

    Called once for calls and once for puts.  Working month by month keeps memory small: a month
    of start dates only needs itself plus the following month (two months for h > 10) as the
    forward side.  The result is identical to a single big merge.
    """
    h = cfg.horizon
    ahead = 1 if h <= 10 else 2
    left = left[PAIR_COLUMNS].copy()
    left["date_next"] = shift_trading_days(left["date"], cal, h)
    pool = pd.concat([f[KEY + list(NEXT_COLUMNS)] for f in (left, nxt) if not f.empty], ignore_index=True)
    pool_period = pool["date"].dt.to_period("M")
    left_period = left["date"].dt.to_period("M")

    chunks = []
    for m in sorted(left_period.unique()):
        lm = left[left_period == m]
        rm = pool[(pool_period >= m) & (pool_period <= m + ahead)].rename(columns=NEXT_COLUMNS)
        chunks.append(lm.merge(rm, on=KEY + ["date_next"], how="inner"))
    pairs = pd.concat(chunks, ignore_index=True)
    del left, pool, chunks

    # --- the target ----------------------------------------------------------------------- #
    pairs["dV"] = pairs["mid_next"] - pairs["mid"]
    pairs["dS"] = pairs["S_next"] - pairs["S"]
    pairs = pairs[pairs["dS"].abs() > cfg.min_abs_ds].copy()
    pairs["y"] = pairs["dV"] / pairs["dS"]             # realised hedge ratio
    pairs["w"] = pairs["dS"] ** 2                      # WLS weight  (w * (y - delta)^2 = hedging error^2)
    pairs["err_bs"] = pairs["dV"] - pairs["delta"] * pairs["dS"]    # benchmark: implied BS delta
    pairs["d_iv"] = pairs["iv_next"] - pairs["iv"]     # useful for Hull-White style diagnostics
    return pairs


def index_log_return(frames: list[pd.DataFrame], cal: pd.DatetimeIndex) -> pd.Series:
    """Daily index log-return log(S_t / S_{t-1}) on the trading calendar, indexed by t.

    NaN where the previous trading day is absent from the data (a vendor gap), never bridged.
    """
    S_daily = daily_index(pd.concat([f for f in frames if not f.empty], ignore_index=True))
    prev_day = shift_trading_days(pd.Series(S_daily.index, index=S_daily.index), cal, -1)
    S_prev = S_daily.reindex(prev_day.values).to_numpy()
    return pd.Series(np.log(S_daily.to_numpy() / S_prev), index=S_daily.index)


def assign_splits(pairs: pd.DataFrame, splits: dict) -> None:
    """Add a ``split`` column in place from year ranges; rows outside every range get 'unused'."""
    year = pairs["date"].dt.year
    split = pd.Series("unused", index=pairs.index)
    for name, (y0, y1) in splits.items():
        split[(year >= y0) & (year <= y1)] = name
    pairs["split"] = split.astype("category")


def build_pairs_year(year: int, cfg: BuildConfig, cal: pd.DatetimeIndex, vix: pd.Series | None,
                     splits: dict, out_path: Path) -> dict:
    """Build and WRITE the pairs whose start date is in ``year``; returns a summary row.

    Calls and puts are built and written one after the other, so at most half a year of
    pairs is in memory at any time.  The end date of a pair may fall in the next year.
    """
    h = cfg.horizon
    left = _read_year(cfg.out_dir, year)
    # the first weeks of next year are needed as the "t+h" side for late-December quotes
    nxt = _read_year(cfg.out_dir, year + 1, months=range(1, 3))
    # the last days of the previous year are needed for the index return at t (t-1 may be in Dec)
    prv = _read_year(cfg.out_dir, year - 1, months=range(12, 13))
    if left.empty:
        return {"year": year, "pairs": 0}
    index_ret = index_log_return([prv, left], cal)
    del prv

    info = {"year": year, "horizon": h, "option_days": int(len(left)), "start_days": int(left["date"].nunique()),
            "pairs": 0, "calls": 0, "puts": 0, "sse_calls": 0.0, "sse_puts": 0.0, "abs_dS_sum": 0.0}
    paired_days: set = set()
    writer = None
    for side, label in (("C", "calls"), ("P", "puts")):
        pairs = _pair_one_side(left[left["type"] == side], nxt[nxt["type"] == side], cal, cfg)

        # --- analysis columns from the paper ---------------------------------------------- #
        pairs["type"] = pairs["type"].astype("category")
        pairs["year"] = pairs["date"].dt.year
        pairs["delta_bucket"] = delta_bucket(pairs["delta"])
        pairs["ttm_class"] = ttm_class(pairs["dte"])             # categorical

        # --- sentiment features (Chen & Li 2023), observed at t, never after --------------- #
        pairs["index_ret"] = pairs["date"].map(index_ret)
        pairs["vix"] = pairs["date"].map(vix) if vix is not None else np.nan
        assign_splits(pairs, splits)

        table = pa.Table.from_pandas(pairs, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

        info["pairs"] += len(pairs); info[label] = int(len(pairs))
        info[f"sse_{label}"] = float((pairs["err_bs"] ** 2).sum())
        info["abs_dS_sum"] += float(pairs["dS"].abs().sum())
        paired_days |= set(pairs["date"].unique())
        del pairs, table
    writer.close()

    # --- bookkeeping ---------------------------------------------------------------------- #
    n = max(info["pairs"], 1)
    info.update({
        "days_with_pairs": len(paired_days),
        "days_lost_to_gaps": info["start_days"] - len(paired_days),
        "pairs_per_day": round(info["pairs"] / max(len(paired_days), 1)),
        "mse_bs_calls": info.pop("sse_calls") / max(info["calls"], 1),
        "mse_bs_puts": info.pop("sse_puts") / max(info["puts"], 1),
        "mean_abs_dS": info.pop("abs_dS_sum") / n,
    })
    return info


def load_vix(path: Path | None) -> pd.Series | None:
    """FRED VIXCLS.csv -> Series indexed by date (level at t).  Returns None if no file."""
    if path is None or not Path(path).exists():
        return None
    v = pd.read_csv(path)
    v.columns = [c.lower() for c in v.columns]
    date_col = "observation_date" if "observation_date" in v.columns else "date"
    v[date_col] = pd.to_datetime(v[date_col])
    v["vixcls"] = pd.to_numeric(v["vixcls"], errors="coerce")
    return v.set_index(date_col)["vixcls"]


def _parts_dir(cfg: BuildConfig) -> Path:
    return cfg.out_dir / f"pairs_h{cfg.horizon}_parts"


def _single_path(cfg: BuildConfig) -> Path:
    return cfg.out_dir / ("pairs.parquet" if cfg.horizon == 1 else f"pairs_h{cfg.horizon}.parquet")


def run_stage2(cfg: BuildConfig, cal: pd.DatetimeIndex, merge: bool = True) -> pd.DataFrame:
    """Build pairs year by year (one Parquet per year), then merge them into one file."""
    years = cfg.years or sorted({int(p.stem[8:12]) for p in cfg.out_dir.glob("spx_eod_*.parquet")})
    splits = SPLIT_SETS[cfg.splits]
    vix = load_vix(cfg.vix_csv)
    parts_dir = _parts_dir(cfg)
    parts_dir.mkdir(parents=True, exist_ok=True)

    infos = []
    t0 = time.time()
    for y in years:
        info = build_pairs_year(y, cfg, cal, vix, splits, parts_dir / f"year={y}.parquet")
        if info["pairs"] == 0:
            print(f"  {y}: no data"); continue
        infos.append(info)
        print(f"  {y}: {info['pairs']:>9,} pairs  ({info['pairs_per_day']:,}/day, "
              f"{info['days_lost_to_gaps']} start days without a forward quote)  "
              f"MSE_BS calls={info['mse_bs_calls']:.3f} puts={info['mse_bs_puts']:.3f}")
    summary = pd.DataFrame(infos)
    # keep one summary row per year across partial runs
    sum_path = cfg.out_dir / f"pairs_h{cfg.horizon}_summary.csv"
    if sum_path.exists() and not summary.empty:
        old = pd.read_csv(sum_path)
        summary = pd.concat([old[~old["year"].isin(summary["year"])], summary], ignore_index=True)
    summary = summary.sort_values("year").reset_index(drop=True)
    summary.to_csv(sum_path, index=False)
    (cfg.out_dir / f"pairs_h{cfg.horizon}_meta.json").write_text(json.dumps({
        "horizon_trading_days": cfg.horizon, "splits": cfg.splits, "split_years": splits,
        "filters": cfg.filters.__dict__, "vix": str(cfg.vix_csv) if cfg.vix_csv else None,
        "features": FEATURES, "sentiment": SENTIMENT, "key": KEY,
        "built": time.strftime("%Y-%m-%d %H:%M"),
    }, indent=2))
    print(f"stage 2: {int(summary['pairs'].sum()):,} pairs in {parts_dir} ({time.time() - t0:.0f}s)")
    if merge:
        run_merge(cfg)
    return summary


def run_merge(cfg: BuildConfig) -> Path:
    """Stream every year part into one pairs.parquet without loading all years into memory."""
    parts = sorted(_parts_dir(cfg).glob("year=*.parquet"))
    if not parts:
        raise SystemExit(f"no year parts in {_parts_dir(cfg)}; run --stage pairs first")
    single = _single_path(cfg)
    writer, n = None, 0
    t0 = time.time()
    for p in parts:
        table = pq.read_table(p)
        if writer is None:
            writer = pq.ParquetWriter(single, table.schema)
        writer.write_table(table)
        n += table.num_rows
    writer.close()
    print(f"merge: {n:,} pairs from {len(parts)} years -> {single} ({time.time() - t0:.0f}s)")
    return single


# --------------------------------------------------------------------------- #
# 4. Command line                                                             #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "clean", "pairs", "merge"], default="all",
                    help="all = clean + pairs + merge; merge = only stitch existing year parts into pairs.parquet")
    ap.add_argument("--no-merge", action="store_true", help="with --stage pairs: write year parts only")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--years", nargs="*", type=int, help="e.g. --years 2019 2020 (default: all found)")
    ap.add_argument("--horizon", type=int, default=1, help="hedging interval in trading days (1, 5, 21)")
    ap.add_argument("--splits", choices=list(SPLIT_SETS), default="modern")
    ap.add_argument("--traded-only", action="store_true", help="keep only quotes with volume > 0 (the paper's sample)")
    ap.add_argument("--min-dte", type=float, default=14.0)
    ap.add_argument("--min-delta", type=float, default=0.05)
    ap.add_argument("--max-delta", type=float, default=0.95)
    ap.add_argument("--vix", default=None, help="FRED VIXCLS.csv (adds the Chen-Li VIX feature)")
    a = ap.parse_args(argv)

    cfg = BuildConfig(
        raw_dir=Path(a.raw_dir), out_dir=Path(a.out), years=a.years, horizon=a.horizon, splits=a.splits,
        filters=L.FilterConfig(min_dte=a.min_dte, min_abs_delta=a.min_delta, max_abs_delta=a.max_delta,
                               require_volume=a.traded_only),
        vix_csv=Path(a.vix) if a.vix else None,
    )
    cal = trading_days()
    if a.stage in ("all", "clean"):
        run_stage1(cfg, cal)
    if a.stage in ("all", "pairs"):
        run_stage2(cfg, cal, merge=not a.no_merge)
    if a.stage == "merge":
        run_merge(cfg)


if __name__ == "__main__":
    main()
