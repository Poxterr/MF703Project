"""The pipeline's output schemas, written down once and asserted at the end of each stage.

Two tables leave the pipeline:

* the **option-day table** (stage 1, one row per option per quote date), and
* the **pairs table** (stage 2, one row per option paired with itself ``h`` trading days later).

Every loader, whatever the vendor, must produce the option-day schema; every pairing run must
produce the pairs schema.  ``assert_schema`` is called by ``build_dataset`` after each stage, so a
new vendor mapping or a code change that drops or retypes a column fails immediately, with the
column named, instead of surfacing weeks later as a confusing model error.

Also home to the identifiers shared across modules: the contract key, the model features and
the sentiment features, so ``models.py``, ``costs.py`` and ``evaluate.py`` import them from one
place.
"""
from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------- #
# Shared identifiers                                                          #
# --------------------------------------------------------------------------- #
KEY = ["type", "expiry", "strike"]                     # identifies one contract
FEATURES = ["T", "delta", "moneyness", "iv", "theta", "vega", "gamma"]   # Qiao & Wan (2024), Table 2
SENTIMENT = ["index_ret", "vix"]                       # Chen & Li (2023) extras

# --------------------------------------------------------------------------- #
# Stage 1: option-day table                                                   #
# --------------------------------------------------------------------------- #
# name -> pandas dtype kind accepted ("datetime", "float", "str").  Order is the column order.
OPTION_DAY_SCHEMA: dict[str, str] = {
    "date": "datetime",       # quote date
    "expiry": "datetime",     # expiry date
    "type": "str",            # "C" or "P"
    "strike": "float",
    "S": "float",             # underlying close
    "dte": "float",           # calendar days to expiry as the vendor reports it
    "T": "float",             # years to expiry = dte / 365
    "bid": "float", "ask": "float", "mid": "float",
    "last": "float", "volume": "float",
    "bid_size": "float", "ask_size": "float",
    "iv": "float",            # implied volatility, decimal (0.15 = 15%)
    "delta": "float", "gamma": "float",
    "vega": "float",          # per UNIT of volatility (Black-Scholes convention), not per 1%
    "theta": "float",         # per calendar day
    "rho": "float",
    "moneyness": "float",     # S / K
}
OPTION_DAY_COLUMNS = list(OPTION_DAY_SCHEMA)

# --------------------------------------------------------------------------- #
# Stage 2: pairs table                                                        #
# --------------------------------------------------------------------------- #
PAIRS_SCHEMA: dict[str, str] = {
    # identity and day-t state
    "type": "str", "expiry": "datetime", "strike": "float",
    "date": "datetime", "S": "float", "dte": "float", "T": "float",
    "bid": "float", "ask": "float", "mid": "float", "volume": "float",
    "iv": "float", "delta": "float", "gamma": "float", "vega": "float", "theta": "float", "rho": "float",
    "moneyness": "float",
    # day t+h
    "date_next": "datetime", "mid_next": "float", "S_next": "float", "iv_next": "float", "delta_next": "float",
    # target
    "dV": "float",            # mid_next - mid
    "dS": "float",            # S_next - S
    "y": "float",             # dV / dS, the realised hedge ratio
    "w": "float",             # dS^2, WLS weight
    "err_bs": "float",        # dV - delta * dS, the Black-Scholes benchmark error
    "d_iv": "float",          # iv_next - iv
    # analysis columns
    "year": "int",
    "delta_bucket": "float",  # signed, paper's convention: 0.1 ... 0.9 for calls, -0.1 ... -0.9 for puts
    "ttm_class": "str",       # categorical: 0-1m, 1-3m, 3-6m, 6m-1y, 1-2y, >2y
    "index_ret": "float",     # log(S_t / S_{t-1})
    "vix": "float",           # VIX level at t (NaN when no VIX file supplied)
    "split": "str",           # categorical: train / val / test / unused
}
PAIRS_COLUMNS = list(PAIRS_SCHEMA)

_KIND_CHECK = {
    "datetime": pd.api.types.is_datetime64_any_dtype,
    "float": pd.api.types.is_float_dtype,
    "int": pd.api.types.is_integer_dtype,
    "str": lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)
    or isinstance(s, pd.CategoricalDtype),
}


class SchemaError(ValueError):
    pass


def assert_schema(df: pd.DataFrame, schema: dict[str, str], name: str = "table",
                  allow_extra: bool = True) -> None:
    """Raise ``SchemaError`` naming every missing, extra (if not allowed) or mistyped column."""
    problems = []
    missing = [c for c in schema if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    if not allow_extra:
        extra = [c for c in df.columns if c not in schema]
        if extra:
            problems.append(f"unexpected columns: {extra}")
    for col, kind in schema.items():
        if col in df.columns and not _KIND_CHECK[kind](df[col].dtype):
            problems.append(f"column {col!r} has dtype {df[col].dtype}, expected {kind}")
    if problems:
        raise SchemaError(f"{name}: " + "; ".join(problems))


def conform(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Reorder to the schema's column order, keeping any extra columns at the end."""
    extra = [c for c in df.columns if c not in schema]
    return df[[c for c in schema if c in df.columns] + extra]
