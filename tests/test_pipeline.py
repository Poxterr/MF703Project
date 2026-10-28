"""Smoke tests on a tiny synthetic vendor file.  Run with ``pytest tests/``."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import load, pairs as pairs_mod, models, evaluate  # noqa: E402
from src.bs import bs_delta, bs_price, implied_vol  # noqa: E402

HEADER = ("[QUOTE_UNIXTIME], [QUOTE_READTIME], [QUOTE_DATE], [QUOTE_TIME_HOURS], [UNDERLYING_LAST], "
          "[EXPIRE_DATE], [EXPIRE_UNIX], [DTE], [C_DELTA], [C_GAMMA], [C_VEGA], [C_THETA], [C_RHO], "
          "[C_IV], [C_VOLUME], [C_LAST], [C_SIZE], [C_BID], [C_ASK], [STRIKE], [P_BID], [P_ASK], "
          "[P_SIZE], [P_LAST], [P_DELTA], [P_GAMMA], [P_VEGA], [P_THETA], [P_RHO], [P_IV], [P_VOLUME], "
          "[STRIKE_DISTANCE], [STRIKE_DISTANCE_PCT]")


def _row(date, S, expiry, dte, K, cbid, cask, cdelta, civ, pbid, pask, pdelta, piv, cvol=""):
    return (f"0, {date} 16:00, {date}, 16.0, {S}, {expiry}, 0, {dte}, {cdelta}, 0.001, 2.0, -0.1, 0.1, "
            f"{civ}, {cvol}, 0, 1 x 2, {cbid}, {cask}, {K}, {pbid}, {pask}, 3 x 4, 0, {pdelta}, 0.001, "
            f"2.0, -0.1, -0.1, {piv}, , 0, 0")


@pytest.fixture
def raw_file(tmp_path):
    lines = [HEADER,
             _row("2016-01-04", 2000, "2016-02-19", 46, 2000, 40.0, 41.0, 0.52, 0.15, 38.0, 39.0, -0.48, 0.16, 10),
             _row("2016-01-04", 2000, "2016-02-19", 46, 2100, 5.0, 5.5, 0.12, 0.14, 0.0, 0.1, -0.97, 0.3),  # put: zero bid, deep delta
             _row("2016-01-04", 2000, "2016-01-08", 4, 2000, 15.0, 15.5, 0.51, 0.15, 14.0, 14.5, -0.49, 0.15),  # DTE < 14
             _row("2016-01-05", 2010, "2016-02-19", 45, 2000, 46.0, 47.0, 0.56, 0.15, 33.0, 34.0, -0.44, 0.16),
             _row("2016-01-05", 2010, "2016-02-19", 45, 2100, 7.0, 7.5, 0.15, 0.14, 0.0, 0.1, -0.96, 0.3),
             ]
    p = tmp_path / "spx_eod_201601.txt"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_loader_schema_and_filters(raw_file):
    df = load.load_chain_file(raw_file)
    assert list(df.columns) == load.STANDARD_COLUMNS
    # kept: 2000C/2000P on both days, 2100C on both days; dropped: 2100P (zero bid, delta -0.97), DTE=4 rows
    assert len(df) == 6
    assert set(df["type"]) == {"C", "P"}
    assert (df["dte"] >= 14).all() and (df["bid"] > 0).all()
    assert df["bid_size"].iloc[0] in (1.0, 3.0)
    assert np.isclose(df.loc[(df.type == "C") & (df.strike == 2000) & (df.date == "2016-01-04"), "mid"].item(), 40.5)


def test_pairs_target(raw_file):
    df = load.load_chain_file(raw_file)
    p = pairs_mod.assign_splits(pairs_mod.build_pairs(df))
    # three contracts appear on both days
    assert len(p) == 3
    row = p[(p.type == "C") & (p.strike == 2000)].iloc[0]
    assert np.isclose(row["dV"], 46.5 - 40.5) and np.isclose(row["dS"], 10.0)
    assert np.isclose(row["y"], 0.6) and np.isclose(row["w"], 100.0)
    assert np.isclose(row["err_bs"], 6.0 - 0.52 * 10)
    assert (p["split"] == "train").all()


def test_hull_white_and_gain_ratio():
    rng = np.random.default_rng(0)
    n = 5000
    delta = rng.uniform(0.1, 0.9, n)
    S, T, vega = 2000.0, 0.2, 300.0
    dS = rng.normal(0, 20, n)
    # true correction: (0.1 - 0.3 delta + 0.2 delta^2) * vega / (S sqrt T)
    corr = (0.1 - 0.3 * delta + 0.2 * delta**2) * vega / (S * np.sqrt(T))
    dV = (delta + corr) * dS + rng.normal(0, 0.02, n)
    p = pd.DataFrame({"type": "C", "delta": delta, "vega": vega, "S": S, "T": T,
                      "dV": dV, "dS": dS, "y": dV / dS, "w": dS**2})
    hw = models.HullWhite().fit(p)
    assert np.allclose(hw.coef_["C"], [0.1, -0.3, 0.2], atol=0.02)
    p["delta_hw"] = hw.predict(p)
    g = evaluate.gain_ratio(p, "hw")
    assert 0.9 < g <= 1.0
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
