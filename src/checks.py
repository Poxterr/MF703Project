"""Hard input checks: things that must stop the pipeline rather than be reported.

The quality report (``load_report.csv``, ``data_quality.md``) *describes* the data: missing
days, zero-bid days, crossed quotes.  The checks here *reject* input that would silently
corrupt every downstream model if it got through:

* a file whose columns do not match the declared vendor (wrong vendor, wrong product);
* a file whose quote dates fall outside the month its name claims;
* a month with no trading days left after the calendar filter;
* Greeks in the wrong units - the vendor's delta must agree with the delta recomputed from
  its own implied volatility, and its (rescaled) vega with the Black-Scholes formula.

The units check exists because of a real incident: this vendor quotes vega per 1% of
volatility, 100x smaller than the textbook convention.  Without rescaling, the Hull-White
regression still runs and still produces a gain ratio; only its coefficients are 100x off.
Nothing downstream would have flagged it.

Each check raises ``InputError`` with a message that names the file and the problem.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .bs import bs_delta, bs_vega


class InputError(ValueError):
    pass


def check_vendor_columns(raw: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(raw.columns))
    if missing:
        raise InputError(f"{path.name}: not in the expected vendor layout; missing columns {missing[:6]}"
                         f"{' ...' if len(missing) > 6 else ''}. Wrong --vendor, or a different product?")


def check_dates_match_filename(long: pd.DataFrame, path: Path) -> None:
    """``spx_eod_201903.txt`` must contain only March 2019 quote dates."""
    ym = path.stem[-6:]
    if not ym.isdigit():
        return  # file not named by month; nothing to compare against
    y, m = int(ym[:4]), int(ym[4:])
    bad = long.loc[(long["date"].dt.year != y) | (long["date"].dt.month != m), "date"]
    if len(bad):
        raise InputError(f"{path.name}: {len(bad)} rows dated outside {y}-{m:02d} "
                         f"(e.g. {bad.iloc[0].date()}). File misnamed or concatenated?")


def check_not_empty(df: pd.DataFrame, path: Path, stage: str) -> None:
    if df.empty or df["date"].nunique() == 0:
        raise InputError(f"{path.name}: no rows left after {stage}. Empty file, or every day fell outside "
                         f"the exchange calendar?")


def check_greek_units(long: pd.DataFrame, path: Path, r: float = 0.01, q: float = 0.018,
                      sample: int = 5000, seed: int = 0,
                      delta_tol: float = 0.05, vega_ratio_range: tuple[float, float] = (0.7, 1.4)) -> dict:
    """Vendor delta and vega must agree with Black-Scholes recomputed from the vendor's own IV.

    Tolerances are loose on purpose: we do not know the vendor's exact rate or dividend yield,
    so recomputed Greeks differ by a few percent.  A factor-of-100 units error, a vol quoted in
    percent instead of decimal, or a delta column swapped with another all fail by a mile.
    Returns the measured agreement so the report can record it.
    """
    liquid = long[(long["bid"] > 0) & (long["dte"] >= 14) & long["delta"].abs().between(0.2, 0.8)]
    if len(liquid) < 200:
        return {"units_check": "skipped (too few liquid rows)"}
    med_iv = float(liquid["iv"].median())
    if not (0.03 <= med_iv <= 1.5):
        raise InputError(f"{path.name}: median implied vol of liquid options is {med_iv:.3g}; expected a "
                         f"decimal around 0.1-0.4. IV quoted in percent?")
    s = liquid[liquid["iv"].between(0.03, 1.5)]
    s = s.sample(min(sample, len(s)), random_state=seed)
    kinds = s["type"].to_numpy()
    d_re = bs_delta(s["S"], s["strike"], s["T"], r, s["iv"], kinds, q)
    v_re = bs_vega(s["S"], s["strike"], s["T"], r, s["iv"], q)
    delta_mad = float(np.median(np.abs(d_re - s["delta"].to_numpy())))
    vega_ratio = float(np.median(s["vega"].to_numpy() / v_re))
    if delta_mad > delta_tol:
        raise InputError(f"{path.name}: vendor delta disagrees with Black-Scholes delta from the vendor's IV "
                         f"(median |diff| = {delta_mad:.3f}). IV in percent? Delta column swapped?")
    lo, hi = vega_ratio_range
    if not (lo <= vega_ratio <= hi):
        raise InputError(f"{path.name}: vega is off by a factor of {vega_ratio:.3g} versus Black-Scholes "
                         f"(after VENDOR_UNITS scaling). Check VENDOR_UNITS['<vendor>']['vega_scale'].")
    return {"delta_median_abs_diff": round(delta_mad, 4), "vega_ratio_to_bs": round(vega_ratio, 3)}
