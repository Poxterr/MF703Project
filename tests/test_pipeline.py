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

from src import load, build_dataset as B, models, evaluate, checks, schema, store  # noqa: E402
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


def test_invalid_greeks_rows_are_dropped(raw_dir):
    """A put with the vendor's negative-vega glitch must not reach the modelling table."""
    p = raw_dir / "2016" / "spx_eod_201601.txt"
    long = load.to_long(load.read_raw(p))
    clean = load.apply_filters(long)
    victim = long[(long["type"] == "P") & (long["strike"] == 2000) & (long["date"] == "2016-01-05")].index
    long.loc[victim, "vega"] = -105449.0                      # as found in spx_eod_201806.txt
    assert not load.greeks_valid(long).loc[victim].any()
    assert len(load.apply_filters(long)) == len(clean) - 1
    off = load.FilterConfig(require_valid_greeks=False)
    assert len(load.apply_filters(long, off)) == len(clean)   # the switch works


def test_corrupt_iv_day_is_dropped_and_reported(raw_dir, cal):
    """A day whose put IVs are ~0 (as on 2019-05-02) is removed whole and listed in the report."""
    strikes = tuple(range(1950, 2100, 5))                      # enough near-the-money rows to judge the day
    lines = [HEADER]
    for d in JAN:
        dte = (pd.Timestamp("2016-03-18") - pd.Timestamp(d)).days
        for K in strikes:
            row = _row(d, S_JAN[d], "2016-03-18", dte, K)
            if d == "2016-01-12":                              # break the put IV on this one day
                f = row.split(", "); f[29] = "0.00050"; row = ", ".join(f)
            lines.append(row)
    p = raw_dir / "2016" / "spx_eod_201601.txt"
    p.write_text("\n".join(lines) + "\n")
    long = load.to_long(load.read_raw(p))
    assert load.corrupt_iv_days(long) == [pd.Timestamp("2016-01-12")]
    cfg = B.BuildConfig(raw_dir=raw_dir, out_dir=raw_dir.parent / "out")
    df, rep = B.clean_file(p, cfg, cal)
    assert pd.Timestamp("2016-01-12") not in set(df["date"])
    assert rep["corrupt_iv_dates"] == "2016-01-12" and rep["corrupt_iv_rows_dropped"] == len(strikes) * 2
    assert "2016-01-12" not in str(rep["missing_trading_days"])
    pairs = B.pair_frame(df, cal)
    assert "2016-01-11" not in set(pairs["date"].dt.strftime("%Y-%m-%d"))   # no forward leg, nothing bridged


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


def _synthetic_pairs(n=6000, seed=0):
    """A pairs-like frame where the true hedge ratio is delta - 0.05 for calls and delta - 0.02 for
    puts, and the option price also moves with an independent IV shock."""
    rng = np.random.default_rng(seed)
    kind = np.where(rng.random(n) < 0.5, "C", "P")
    mag = rng.uniform(0.1, 0.9, n)
    delta = np.where(kind == "C", mag, -mag)
    dates = pd.to_datetime("2018-01-02") + pd.to_timedelta(rng.integers(0, 1080, n), unit="D")
    p = pd.DataFrame({
        "type": pd.Categorical(kind), "expiry": pd.Timestamp("2022-01-21"),
        "strike": rng.permutation(n).astype("float64"), "date": dates,
        "T": rng.uniform(0.05, 1, n), "delta": delta, "moneyness": rng.uniform(0.9, 1.1, n),
        "iv": rng.uniform(0.1, 0.3, n), "theta": -rng.uniform(0, 1, n),
        "vega": rng.uniform(100, 400, n), "gamma": rng.uniform(0, 0.01, n), "S": 3000.0,
        "dte": rng.uniform(14, 400, n),
    })
    p["year"] = p["date"].dt.year
    p["split"] = np.select([p.year == 2018, p.year == 2019], ["train", "val"], "test")
    p["dS"] = rng.normal(0, 20, n)
    p["d_iv"] = rng.normal(0, 0.004, n)
    shift = np.where(kind == "C", -0.05, -0.02)
    p["dV"] = (p["delta"] + shift) * p["dS"] + p["vega"] * p["d_iv"] + rng.normal(0, 0.2, n)
    p["y"], p["w"] = p["dV"] / p["dS"], p["dS"] ** 2
    p["delta_bucket"] = B.delta_bucket(p["delta"]); p["ttm_class"] = B.ttm_class(p["dte"])
    return p


def test_gbm_residual_learns_the_shift():
    pytest.importorskip("lightgbm")
    p = _synthetic_pairs()
    tr, va, te = (p[p.split == s] for s in ("train", "val", "test"))
    gbm = models.GBMResidual(params=dict(n_estimators=200, learning_rate=0.1, num_leaves=7,
                                         min_child_samples=50, subsample=0.8, subsample_freq=1,
                                         verbose=-1)).fit(tr, val=va)
    te = evaluate.add_predictions(te, {"gbm": gbm})
    assert evaluate.gain_ratio(te, "gbm") > 0.0                 # beats Black-Scholes out of sample
    corr = (te["delta_gbm"] - te["delta"]).groupby(te["type"], observed=True).mean()
    assert abs(corr["C"] + 0.05) < 0.02 and abs(corr["P"] + 0.02) < 0.02
    assert "is_call" in gbm.importance().index
    # same seed -> same model
    again = models.GBMResidual(params=gbm.params).fit(tr, val=va)
    assert np.allclose(again.predict(te), gbm.predict(te))


def test_standard_report_tables_and_no_mutation(tmp_path):
    p = _synthetic_pairs()
    p["delta_m"] = p["delta"] + np.where(p["type"] == "C", -0.05, -0.02)     # the true hedge
    cols_before = list(p.columns)
    tables = evaluate.standard_report(p, ["m"], tmp_path / "t")
    assert list(p.columns) == cols_before                                    # nothing added to the big table
    assert set(tables) == {"by_split_type", "by_year_type", "by_delta_bucket", "by_maturity", "crisis"}
    assert tables["crisis"]["n"].sum() == evaluate.crisis_mask(p).sum()
    assert (tables["by_split_type"]["m"] > 0).all()
    assert tables["by_split_type"]["n"].sum() == len(p)
    assert tables["by_delta_bucket"]["n"].sum() == (p.split == "test").sum()   # test split only
    assert (tmp_path / "t" / "gain_by_year_type.csv").exists()
    # the grouped numbers agree with the direct formula
    sub = p[(p.split == "test") & (p.type == "C")]
    assert np.isclose(tables["by_split_type"].loc[("test", "C"), "m"], evaluate.gain_ratio(sub, "m"))
    # no crisis table when the window is empty
    q = p.copy(); q["date"] = pd.Timestamp("2019-06-03")
    assert "crisis" not in evaluate.standard_report(q, ["m"])


def test_vega_attribution_and_ceiling():
    p = _synthetic_pairs()
    va = evaluate.vega_attribution(p, ["type"])
    assert (va["vega_share"] > 0.5).all() and (va["vega_share"] < 1).all()   # IV shocks are most of the error
    # if dV has no IV component, attributing one only adds error: the share goes negative
    q = p.assign(dV=p["dV"] - p["vega"] * p["d_iv"])
    assert evaluate.vega_attribution(q)["vega_share"].iloc[0] < 0
    ceil = evaluate.in_sample_ceiling(p, models.HullWhite, "hw")
    assert list(ceil.columns) == ["hw_ceiling"] and len(ceil) == p["year"].nunique() * 2
    hw = models.HullWhite().fit(p[p.year == 2018])
    oos = evaluate.gain_ratio(evaluate.add_predictions(p[(p.year == 2020) & (p.type == "C")], {"hw": hw}), "hw")
    assert ceil.loc[(2020, "C"), "hw_ceiling"] >= oos - 1e-9                 # in-sample fit is an upper bound


def test_predictions_roundtrip_and_mismatch(tmp_path):
    p = _synthetic_pairs(500)
    p["delta_hw"] = (p["delta"] - 0.03).astype("float32")
    store.save_predictions(p, ["hw"], tmp_path / "pred")
    back = store.load_predictions(p.drop(columns="delta_hw").sample(frac=1, random_state=1), ["hw"], tmp_path / "pred")
    merged = back.set_index(store.ROW_KEY)["delta_hw"].sort_index()
    assert np.allclose(merged, p.set_index(store.ROW_KEY)["delta_hw"].sort_index())
    with pytest.raises(FileNotFoundError, match="run the notebook"):
        store.load_predictions(p, ["nn_resid"], tmp_path / "pred")
    other = _synthetic_pairs(500, seed=9)                                    # a different build
    with pytest.raises(ValueError, match="different build"):
        store.load_predictions(other, ["hw"], tmp_path / "pred")


def test_load_pairs_is_lean(tmp_path):
    p = _synthetic_pairs(300)
    p["extra_wide_column"] = 1.0
    p.to_parquet(tmp_path / "pairs.parquet", index=False)
    cols = [c for c in store.MODEL_COLUMNS if c in p.columns]
    df = store.load_pairs(tmp_path / "pairs.parquet", columns=cols, years=[2020])
    assert "extra_wide_column" not in df and set(df["year"]) == {2020}
    assert df["dS"].dtype == "float32" and df["date"].dtype.kind == "M"


def test_sample_paths_are_separate():
    import config
    a, b = config.sample_paths("main"), config.sample_paths("traded_only")
    assert a["pairs"] != b["pairs"] and a["tables"] != b["tables"] and a["predictions"] != b["predictions"]
    assert a["config"].exists() and b["config"].exists()
    with pytest.raises(KeyError):
        config.sample_paths("nope")


def test_black_scholes_roundtrip():
    S, K, T, r, sig = 100.0, 105.0, 0.5, 0.02, 0.25
    c = bs_price(S, K, T, r, sig, "C")
    assert np.isclose(implied_vol(np.array([c]), S, K, T, r, "C")[0], sig, atol=1e-6)
    assert 0 < bs_delta(S, K, T, r, sig, "C") < 1
    assert -1 < bs_delta(S, K, T, r, sig, "P") < 0


def test_feature_matrix_derived_features_are_scale_free():
    p = _synthetic_pairs(200)
    X = models.feature_matrix(p, ["T", "log_m", "vega_s", "gamma_s", "theta_s"])
    assert X.dtype == np.float32 and X.shape == (200, 5)
    assert np.allclose(X[:, 2], p["vega"] / p["S"], rtol=1e-5)
    # doubling the index level with moneyness fixed: vega and theta double, gamma halves -> same inputs
    q = p.assign(S=p["S"] * 2, vega=p["vega"] * 2, theta=p["theta"] * 2, gamma=p["gamma"] / 2)
    assert np.allclose(models.feature_matrix(q, ["vega_s", "gamma_s", "theta_s"]), X[:, 2:], rtol=1e-5)
    with pytest.raises(KeyError, match="neither a column"):
        models.feature_matrix(p, ["no_such_feature"])


def test_neural_hedger_options():
    pytest.importorskip("torch")
    p = _synthetic_pairs(3000)
    kw = dict(hidden=(8,), max_epochs=3, patience=3, features=["T", "delta"])
    plain = models.NeuralHedger(residual=True, **kw).fit(p, val=p)
    # chunked scoring gives the same predictions as one pass
    chunked = models.NeuralHedger(residual=True, eval_batch=97, **kw).fit(p, val=p)
    assert np.allclose(plain.predict(p), chunked.predict(p), atol=1e-5)
    # Hull-White scaling: the correction is the network output times vega / (S sqrt(T))
    hw = models.NeuralHedger(residual=True, output_scale="hw", label="nn_hw", **kw).fit(p, val=p)
    assert hw.name == "nn_hw"
    corr = hw.predict(p) - p["delta"].to_numpy()
    q = p.assign(vega=p["vega"] * 3)                       # 3x the vega -> 3x the correction, same inputs
    assert np.allclose(hw.predict(q) - q["delta"].to_numpy(), 3 * corr, rtol=1e-4, atol=1e-6)
    # clipping: inputs far outside the training range are treated as the edge of it
    clip = models.NeuralHedger(residual=True, clip_inputs=2.0, **kw).fit(p, val=p)
    far = p.head(5).assign(T=1e6); edge = p.head(5).assign(T=p["T"].mean() + 2.0 * p["T"].std(ddof=0))
    assert np.allclose(clip.predict(far), clip.predict(edge), atol=1e-4)
    # same seed -> same model; different seed -> different model
    again = models.NeuralHedger(residual=True, **kw).fit(p, val=p)
    other = models.NeuralHedger(residual=True, seed=1, **kw).fit(p, val=p)
    assert np.allclose(plain.predict(p), again.predict(p)) and not np.allclose(plain.predict(p), other.predict(p))
    with pytest.raises(ValueError):
        models.NeuralHedger(residual=False, output_scale="hw")
    with pytest.raises(ValueError):
        models.NeuralHedger(output_scale="nope")


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


# --------------------------------------------------------------------------- #
# Transaction costs                                                           #
# --------------------------------------------------------------------------- #
def _cost_pairs():
    """One call quoted on four days with a gap after the second: two episodes of two days each."""
    d = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-07", "2020-01-08"])
    nxt = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-08", "2020-01-09"])
    return pd.DataFrame({
        "type": "C", "expiry": pd.Timestamp("2020-03-20"), "strike": 3000.0,
        "date": d, "date_next": nxt, "S": [3000.0, 3010.0, 3020.0, 3000.0],
        "dS": [10.0, -5.0, -20.0, 10.0], "dV": [6.0, -2.0, -9.0, 4.0],
        "delta": [0.5, 0.6, 0.55, 0.45],
    })


def test_costs_episodes_split_on_gap():
    from src import costs
    p = costs.sort_for_episodes(_cost_pairs().sample(frac=1, random_state=0))   # order must not matter
    assert p["episode"].tolist() == [0, 0, 1, 1]
    pnl = costs.hedge_pnl(p, ["delta"], cost_bp=0, min_days=1)
    assert pnl["n_days"].tolist() == [2, 2]
    # a subset of episodes (ids no longer 0..k-1) must still work
    only = costs.hedge_pnl(p[p["episode"] == 1], ["delta"], cost_bp=0, min_days=1)
    assert len(only) == 1 and only.loc[0, "pnl_delta"] == pytest.approx(pnl.loc[1, "pnl_delta"])


def test_costs_known_numbers():
    from src import costs
    p = _cost_pairs()
    pnl = costs.hedge_pnl(p, ["delta"], cost_bp=10, min_days=1)        # c = 1e-3
    # episode 0: gross = (-6 + 0.5*10) + (2 + 0.6*-5) = -2 ; trades 0.5 @3000, 0.1 @3010, unwind 0.6 @3005
    assert pnl.loc[0, "pnl_delta"] == pytest.approx(-2.0 - 1e-3 * (0.5 * 3000 + 0.1 * 3010 + 0.6 * 3005))
    assert pnl.loc[0, "turnover_delta"] == pytest.approx(0.5 + 0.1 + 0.6)
    assert pnl.loc[0, "rebal_delta"] == pytest.approx(0.1)
    # episode 1 sets up from zero again: 0.55, then 0.10, then unwind 0.45
    assert pnl.loc[1, "turnover_delta"] == pytest.approx(0.55 + 0.10 + 0.45)
    assert pnl.loc[1, "cost_delta"] == pytest.approx(1e-3 * (0.55 * 3020 + 0.10 * 3000 + 0.45 * 3010))


def test_costs_zero_cost_gain_matches_evaluate():
    from src import costs, evaluate
    p = _cost_pairs()
    p["delta_m"] = p["delta"] - 0.05
    g0 = costs.net_gain(p, ["m"], bps=(0,)).loc[0, "m"]
    assert g0 == pytest.approx(evaluate.gain_ratio(p, "m"))
    by = costs.net_gain(p, ["m"], bps=(0, 5), by=["type"])
    assert by.loc[(0, "C"), "m"] == pytest.approx(g0)
