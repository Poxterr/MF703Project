"""Tests for the data pipeline and models.  Run with ``pytest tests/``.

Everything runs on small synthetic vendor files written to a temp folder, so the suite needs
no real data and finishes in seconds.  The synthetic files follow the optionsDX layout exactly
(bracketed headers, "1 x 2" size strings, blank-string NAs) so the parser is tested for real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import load, build_dataset as B, models, evaluate, checks, schema  # noqa: E402
from src.bs import bs_delta, bs_price, bs_vega, implied_vol  # noqa: E402

HEADER = ("[QUOTE_UNIXTIME], [QUOTE_READTIME], [QUOTE_DATE], [QUOTE_TIME_HOURS], [UNDERLYING_LAST], "
          "[EXPIRE_DATE], [EXPIRE_UNIX], [DTE], [C_DELTA], [C_GAMMA], [C_VEGA], [C_THETA], [C_RHO], "
          "[C_IV], [C_VOLUME], [C_LAST], [C_SIZE], [C_BID], [C_ASK], [STRIKE], [P_BID], [P_ASK], "
          "[P_SIZE], [P_LAST], [P_DELTA], [P_GAMMA], [P_VEGA], [P_THETA], [P_RHO], [P_IV], [P_VOLUME], "
          "[STRIKE_DISTANCE], [STRIKE_DISTANCE_PCT]")

R, Q = 0.01, 0.018   # rate and dividend yield used to generate consistent synthetic Greeks


def _row(date, S, expiry, dte, K, iv=0.15, cvol=10):
    """One vendor row with Black-Scholes-consistent prices and Greeks (vega per 1%, like the vendor)."""
    T = dte / 365
    c = bs_price(S, K, T, R, iv, "C", Q); p = bs_price(S, K, T, R, iv, "P", Q)
    cd = bs_delta(S, K, T, R, iv, "C", Q); pdl = bs_delta(S, K, T, R, iv, "P", Q)
    vega = bs_vega(S, K, T, R, iv, Q) / 100
    return (f"0, {date} 16:00, {date}, 16.0, {S}, {expiry}, 0, {dte}, {cd:.5f}, 0.001, {vega:.5f}, -0.1, 0.1, "
            f"{iv}, {cvol}, 0, 1 x 2, {c - 0.5:.4f}, {c + 0.5:.4f}, {K}, {p - 0.5:.4f}, {p + 0.5:.4f}, 3 x 4, 0, "
            f"{pdl:.5f}, 0.001, {vega:.5f}, -0.1, -0.1, {iv}, , 0, 0")


def write_month(path: Path, dates: list[str], S_by_date: dict[str, float], expiry="2016-03-18",
                strikes=(1900, 1950, 2000, 2050, 2100)) -> Path:
    lines = [HEADER]
    for d in dates:
        dte = (pd.Timestamp(expiry) - pd.Timestamp(d)).days
        for K in strikes:
            lines.append(_row(d, S_by_date[d], expiry, dte, K))
    path.write_text("\n".join(lines) + "\n")
    return path


# January 2016 NYSE trading days: Jan 4-8, 11-15, 19-22 (18th is MLK day), 25-29.
JAN = ["2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07", "2016-01-08",
       "2016-01-11", "2016-01-12", "2016-01-13", "2016-01-14", "2016-01-15",
       "2016-01-19", "2016-01-20", "2016-01-21", "2016-01-22",
       "2016-01-25", "2016-01-26", "2016-01-27", "2016-01-28", "2016-01-29"]
S_JAN = {d: 2000 + 5 * i for i, d in enumerate(JAN)}


@pytest.fixture
def raw_dir(tmp_path):
    """A raw folder with one clean January 2016 file."""
    d = tmp_path / "raw" / "2016"; d.mkdir(parents=True)
    write_month(d / "spx_eod_201601.txt", JAN, S_JAN)
    return tmp_path / "raw"


@pytest.fixture
def cal():
    return B.trading_days("2015-12-01", "2016-12-31")


# --------------------------------------------------------------------------- #
# Stage 1: parsing, filters, schema                                           #
# --------------------------------------------------------------------------- #
def test_loader_schema_and_units(raw_dir):
    df = load.load_chain_file(raw_dir / "2016" / "spx_eod_201601.txt")
    schema.assert_schema(df, schema.OPTION_DAY_SCHEMA, "test", allow_extra=False)
    assert set(df["type"]) == {"C", "P"}
    assert (df["dte"] >= 14).all() and (df["bid"] > 0).all()
    assert df["bid_size"].iloc[0] in (1.0, 3.0)
    # vega must come out in textbook units after VENDOR_UNITS scaling
    row = df.iloc[0]
    assert np.isclose(row["vega"], bs_vega(row["S"], row["strike"], row["T"], R, row["iv"], Q), rtol=0.02)


def test_filters_drop_short_dte_and_zero_bid(tmp_path):
    p = tmp_path / "spx_eod_201603.txt"
    lines = [HEADER,
             _row("2016-03-01", 2000, "2016-03-18", 17, 2000),           # kept
             _row("2016-03-10", 2000, "2016-03-18", 8, 2000),            # dte < 14 -> dropped
             _row("2016-03-01", 2000, "2016-03-18", 17, 2600, iv=0.5)]   # call delta ~0 -> dropped
    p.write_text("\n".join(lines) + "\n")
    df = load.load_chain_file(p)
    assert len(df) == 2 and set(df["type"]) == {"C", "P"}


# --------------------------------------------------------------------------- #
# Hard input checks                                                           #
# --------------------------------------------------------------------------- #
def test_wrong_vendor_columns_rejected(tmp_path):
    p = tmp_path / "spx_eod_201601.txt"
    p.write_text("date,strike,call_bid\n2016-01-04,2000,10\n")
    with pytest.raises(checks.InputError, match="vendor layout"):
        checks.check_vendor_columns(load.read_raw(p), load.required_columns("optionsdx"), p)


def test_misnamed_file_rejected(tmp_path):
    p = write_month(tmp_path / "spx_eod_201602.txt", JAN[:3], S_JAN)   # January dates, February name
    long = load.to_long(load.read_raw(p))
    with pytest.raises(checks.InputError, match="outside 2016-02"):
        checks.check_dates_match_filename(long, p)


def test_units_check_catches_unscaled_vega(raw_dir):
    p = raw_dir / "2016" / "spx_eod_201601.txt"
    long = load.to_long(load.read_raw(p))
    # bulk it up past the 200-row threshold by repeating, then break the units
    long = pd.concat([long] * 10, ignore_index=True)
    assert "vega_ratio_to_bs" in checks.check_greek_units(long, p)          # correct units pass
    bad = long.assign(vega=long["vega"] / 100)                               # the vendor's raw convention
    with pytest.raises(checks.InputError, match="vega is off by a factor"):
        checks.check_greek_units(bad, p)
    bad = long.assign(iv=long["iv"] * 100)                                   # IV in percent
    with pytest.raises(checks.InputError):
        checks.check_greek_units(bad, p)


def test_schema_assertion_names_problem():
    df = pd.DataFrame({"date": pd.to_datetime(["2016-01-04"]), "strike": ["2000"]})
    with pytest.raises(schema.SchemaError) as e:
        schema.assert_schema(df, {"date": "datetime", "strike": "float", "S": "float"}, "t")
    assert "missing columns: ['S']" in str(e.value) and "'strike' has dtype" in str(e.value)


# --------------------------------------------------------------------------- #
# Stage 2: calendar-correct pairing                                           #
# --------------------------------------------------------------------------- #
def test_pairs_target_arithmetic(raw_dir, cal):
    df = load.load_chain_file(raw_dir / "2016" / "spx_eod_201601.txt")
    pairs = B.pair_frame(df, cal, horizon=1, splits="proposal")
    schema.assert_schema(pairs, schema.PAIRS_SCHEMA, "pairs")
    row = pairs[(pairs["type"] == "C") & (pairs["strike"] == 2000) & (pairs["date"] == "2016-01-04")].iloc[0]
    assert row["date_next"] == pd.Timestamp("2016-01-05")
    assert np.isclose(row["dS"], 5.0)
    assert np.isclose(row["dV"], row["mid_next"] - row["mid"])
    assert np.isclose(row["y"], row["dV"] / row["dS"]) and np.isclose(row["w"], 25.0)
    assert np.isclose(row["err_bs"], row["dV"] - row["delta"] * row["dS"])
    assert (pairs["split"] == "train").all()
    assert set(pairs["delta_bucket"].unique()) <= {x / 10 for x in range(-9, 10) if x}


def test_gap_drops_pair_instead_of_bridging(raw_dir, cal):
    """If Jan 6 is missing from the file, Jan 5 must have NO pair (not a two-day move to Jan 7)."""
    dates = [d for d in JAN if d != "2016-01-06"]
    p = write_month(raw_dir / "2016" / "spx_eod_201601.txt", dates, S_JAN)
    pairs = B.pair_frame(load.load_chain_file(p), cal)
    assert "2016-01-05" not in set(pairs["date"].dt.strftime("%Y-%m-%d"))
    assert (pairs["date_next"] - pairs["date"]).dt.days.max() <= 4   # weekends and MLK day only


def test_holiday_rows_are_dropped(raw_dir, cal):
    """Rows dated on a market holiday (MLK day, Jan 18 2016) must be removed, not paired."""
    dates = JAN[:10] + ["2016-01-18"] + JAN[10:]
    S = {**S_JAN, "2016-01-18": 2040.0}
    p = write_month(raw_dir / "2016" / "spx_eod_201601.txt", dates, S)
    cfg = B.BuildConfig(raw_dir=raw_dir, out_dir=raw_dir.parent / "out")
    df, rep = B.clean_file(p, cfg, cal)
    assert rep["non_trading_rows_dropped"] == 10          # 5 strikes x 2 sides
    assert "2016-01-18" in rep["non_trading_dates"]
    assert pd.Timestamp("2016-01-18") not in set(df["date"])
    # Jan 15 pairs with Jan 19 (the real next trading day), a 4-calendar-day move
    pairs = B.pair_frame(df, cal)
    nxt = pairs.loc[pairs["date"] == "2016-01-15", "date_next"].unique()
    assert list(nxt) == [pd.Timestamp("2016-01-19")]


# --------------------------------------------------------------------------- #
# End to end                                                                  #
# --------------------------------------------------------------------------- #
def test_build_dataset_end_to_end(raw_dir, tmp_path):
    out = tmp_path / "processed"
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(f"""
years: [2016]
horizon: 1
raw_dir: {raw_dir}
out_dir: {out}
splits: {{train: [2016, 2016], val: [2017, 2017], test: [2018, 2018]}}
filters: {{min_dte: 14, min_abs_delta: 0.05, max_abs_delta: 0.95}}
""")
    B.main(["--config", str(cfg_file)])
    for f in ["spx_eod_201601.parquet", "load_report.csv", "data_quality.md", "pairs.parquet",
              "pairs_h1_summary.csv", "pairs_h1_meta.json", "build_config.yaml"]:
        assert (out / f).exists(), f
    pairs = pd.read_parquet(out / "pairs.parquet")
    schema.assert_schema(pairs, schema.PAIRS_SCHEMA, "pairs")
    assert 0 < len(pairs) <= 18 * 5 * 2      # at most 18 start days x 5 strikes x 2 sides (delta filter trims a few)
    assert pairs["date"].max() == pd.Timestamp("2016-01-28")    # the last day has no forward quote
    opt = pd.read_parquet(out / "spx_eod_201601.parquet")
    key = ["type", "expiry", "strike", "date"]
    have = set(map(tuple, opt[key].to_numpy()))
    assert all(tuple(r) in have for r in pairs[key].to_numpy())                      # day t exists
    assert all(tuple(r) in have for r in pairs[["type", "expiry", "strike", "date_next"]].to_numpy())  # day t+1 exists
    assert (pairs["split"] == "train").all()
    rep = pd.read_csv(out / "load_report.csv")
    assert rep.loc[0, "trading_days"] == 19 and rep.loc[0, "missing_trading_days"] != rep.loc[0, "missing_trading_days"]  # NaN = none missing
    # a flag overrides the file
    B.main(["--config", str(cfg_file), "--min-dte", "60", "--out", str(tmp_path / "p2")])
    assert not (tmp_path / "p2" / "pairs.parquet").exists() or len(pd.read_parquet(tmp_path / "p2" / "pairs.parquet")) < len(pairs)


# --------------------------------------------------------------------------- #
# Models and evaluation                                                       #
# --------------------------------------------------------------------------- #
def test_hull_white_and_gain_ratio():
    rng = np.random.default_rng(0)
    n = 5000
    delta = rng.uniform(0.1, 0.9, n)
    S, T, vega = 2000.0, 0.2, 300.0
    dS = rng.normal(0, 20, n)
    corr = (0.1 - 0.3 * delta + 0.2 * delta**2) * vega / (S * np.sqrt(T))
    dV = (delta + corr) * dS + rng.normal(0, 0.02, n)
    p = pd.DataFrame({"type": "C", "delta": delta, "vega": vega, "S": S, "T": T,
                      "dV": dV, "dS": dS, "y": dV / dS, "w": dS**2})
    hw = models.HullWhite().fit(p)
    assert np.allclose(hw.coef_["C"], [0.1, -0.3, 0.2], atol=0.02)
    p["delta_hw"] = hw.predict(p)
    assert 0.9 < evaluate.gain_ratio(p, "hw") <= 1.0
    assert np.isclose(evaluate.gain_ratio(p, "bs"), 0.0)


def test_black_scholes_roundtrip():
    S, K, T, r, sig = 100.0, 105.0, 0.5, 0.02, 0.25
    c = bs_price(S, K, T, r, sig, "C")
    assert np.isclose(implied_vol(np.array([c]), S, K, T, r, "C")[0], sig, atol=1e-6)
    assert 0 < bs_delta(S, K, T, r, sig, "C") < 1
    assert -1 < bs_delta(S, K, T, r, sig, "P") < 0


def test_neural_hedger_runs():
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    n = 2000
    p = pd.DataFrame({
        "type": "C", "T": rng.uniform(0.05, 1, n), "delta": rng.uniform(0.1, 0.9, n),
        "moneyness": rng.uniform(0.9, 1.1, n), "iv": rng.uniform(0.1, 0.3, n),
        "theta": -rng.uniform(0, 1, n), "vega": rng.uniform(100, 400, n), "gamma": rng.uniform(0, 0.01, n),
    })
    p["dS"] = rng.normal(0, 20, n)
    p["dV"] = (p["delta"] - 0.05) * p["dS"] + rng.normal(0, 0.5, n)
    p["y"], p["w"] = p["dV"] / p["dS"], p["dS"]**2
    m = models.NeuralHedger(residual=True, hidden=(16,), max_epochs=5, patience=5).fit(p, val=p)
    out = m.predict(p)
    assert out.shape == (n,) and np.isfinite(out).all()
