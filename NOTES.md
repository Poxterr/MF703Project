# Working notes

## 2016 data (loaded 2026-10-28)

- 12 monthly files, 251 trading days, 940,596 strike rows -> 661,539 filtered option rows -> 609,406 next-day pairs.
- Stage 1 runs in ~11 s for the year; stage 2 in a few seconds. Parquet output ~36 MB.
- Vendor vega is per 1% vol; the loader rescales it x100 so Hull-White coefficients are in textbook units
  (calls: a=-0.20, b=0.59, c=-0.79 on Jan-Sep 2016).

## Early finding: the correction is unstable month to month in 2016

Fitting Hull-White on Jan-Sep 2016 and testing on Oct-Dec 2016 gives gain ratios of 0.27 (calls) /
0.19 (puts) in-sample but -0.09 / -0.26 out of sample. The one-parameter correction coefficient is
negative in 10 of 12 months (vol rises when the index falls, as theory says) but ranges from -23 to +10
(vendor units), and December 2016 has almost no IV-price correlation. The election move in November sits
in the test window. Expect:

- gains to be modest and regime-dependent; report them by year and by month, not only overall;
- the 2016-2018 training window to average over regimes better than 9 months did;
- the 2020 test year to be the hard case the proposal says it is.

This is a result to report, not a bug to hide.

## 2010-2023 data loaded and paired (2026-10-31)

- Raw: 168 optionsDX monthly files in `data/raw/<year>/spx_eod_YYYYMM.txt` (5.3 GB, not in git).
  2024 was not downloaded. The 2013 download folder was duplicated byte-for-byte; one copy kept.
- New entry point `src/build_dataset.py` runs stage 1 + stage 2 + merge in one go:
  `python -m src.build_dataset` (all years, 1-day horizon, "modern" splits). See its docstring.
  Needs `pandas_market_calendars` (added to requirements.txt).
- Result: 13,192,861 one-day pairs in `data/processed/pairs.parquet` (803 MB); per-year files in
  `data/processed/pairs_h1_parts/`; counts and benchmark MSE by year in `pairs_h1_summary.csv`;
  every missing/dropped day in `data_quality.md`.
- Vendor problems found and handled (details in data_quality.md):
  * 35 NYSE trading days are absent from the files (e.g. 2011-01-19..25, 2019-03-01, 2022-06-17).
  * 8 days in Jan-Feb 2010 have bid = 0 on every option (filtered out entirely).
  * 2022 files contain rows on all 9 exchange holidays; 2012 has Hurricane Sandy days; 2018 has 5 Dec.
  Pairs are now built on the NYSE calendar: t+1 is the real next trading day, so a gap drops the
  pair instead of silently creating a two-day move (the old `pairs.py` bridged gaps).
- Filter note: with the default bid > 0 filter, calls pile up in the 0.9 delta bucket (deep ITM,
  never traded). The paper's sample is the `--traded-only` run; build both before comparing to it.
- 2020 benchmark MSE (82) is ~20x 2019 (4.6): the crisis year dominates any pooled gain ratio;
  report by year.

## Stage 1 consolidated (2026-11-01)

- One pipeline: `src/build_dataset.py` (Owen's calendar-correct version) is the only loader/pairer;
  `src/pairs.py` is gone. `src/schema.py` holds the two output schemas and the shared identifiers
  (KEY, FEATURES, SENTIMENT); `src/checks.py` holds the hard input checks.
- Builds are driven by YAML configs in `configs/`; `build_config.yaml` is written next to each output.
- Default split is the proposal's (train 2016-18 / val 2019 / test 2020). `paper` and `modern` remain available.
- Full 2016-2020 build: 5,419,731 pairs, 82 s. Numbers and findings in `docs/data_pipeline.md`.

### First baselines on the full 2016-2020 dataset (sanity check, not tuned)

Hull-White and a LightGBM residual model fitted on 2016-18 (GBM early-stopped on 2019), gain ratio
vs the implied Black-Scholes delta:

| | HW calls | HW puts | GBM calls | GBM puts |
|---|---|---|---|---|
| val 2019 | 0.033 | 0.029 | 0.019 | 0.093 |
| test 2020 | 0.190 | 0.036 | 0.164 | 0.107 |
| Feb-Apr 2020 | 0.215 | 0.042 | 0.190 | 0.117 |

Positive out of sample in every year once the training window is three years (the 9-month 2016-only
experiment above went negative). Well below the paper's 0.3-0.6, which used the traded-only sample
and nine training years; run `configs/traded_only_2016_2020.yaml` before comparing.
Hull-White coefficients on the training years: calls (-0.08, -0.15, 0.14), puts (-0.12, -0.37, -0.27).
Memory: load `pairs.parquet` with `columns=` (see `evaluate.EVAL_COLUMNS`) and cast to float32;
the full 35-column table is ~3 GB in pandas.

## Baselines prep (2026-11-20)

Code changes before running notebook 02, and two vendor data defects found while testing them.

- `src/store.py`: `load_pairs` (modelling columns only, float32: 0.54 GB instead of ~3 GB) and
  `save_predictions` / `load_predictions` (one small file per model, joined by contract key and
  date). Notebooks no longer pass full copies of the table to each other.
- `config.sample_paths(SAMPLE)`: inputs and outputs per sample (`main`, `traded_only`), so results
  from the two samples never overwrite each other. Tables go to `tables/<sample>/`.
- `evaluate.py`: no copies of the big table; `standard_report` adds the Feb-Apr 2020 table;
  `vega_attribution` and `in_sample_ceiling` put gain ratios in context.
- `GBMResidual`: row subsampling now active (`subsample_freq`), seed passed, importance().
  `HullWhite` fits in float64.

**Vendor defect 1: impossible put vegas.** About 0.4% of puts have a large negative vega in the raw
files (e.g. -1054.49 against 9.68 for the matching call). `load.greeks_valid` now drops rows with
iv <= 0, vega <= 0, gamma < 0 or vega more than 2x away from Black-Scholes: 16,350 option-days.

**Vendor defect 2: six days of corrupt implied vols.** 2019-05-02, -03, -06, -07, -08, -17: call IV
~0.45, put IV ~0.0005, so every Greek is wrong. `load.corrupt_iv_days` drops them: 65,160 rows.

Dataset is now 5,372,273 pairs (was 5,419,731). **Everyone must rebuild** from
`configs/main_2016_2020.yaml` and check the count against `data/processed/pairs_h1_summary.csv`.

Effect on the untuned baselines (gain ratio vs Black-Scholes, fitted on 2016-18). These replace
the table above; a proper run with commentary belongs in notebook 02.

| | HW calls | HW puts | GBM calls | GBM puts |
|---|---|---|---|---|
| val 2019 | 0.042 | 0.101 | 0.076 | 0.122 |
| test 2020 | 0.191 | 0.161 | 0.156 | 0.067 |
| Feb-Apr 2020 | 0.216 | 0.178 | 0.181 | 0.067 |

Hull-White on puts went from 0.036 to 0.161 on 2020 once the bad vegas were removed.
Hull-White coefficients: calls (-0.082, -0.139, 0.123), puts (-0.247, -0.264, 0.150).

## Baselines run (2026-11-21)

Notebook 02 executed on both samples; outputs in the notebook (main sample), tables in
`tables/<sample>/baselines/`, figures in `figures/<sample>/`. Findings are written at the end of
the notebook. Headline: Hull-White gains 0.042 / 0.101 (calls / puts) on 2019 and 0.191 / 0.161 on
2020, nearly identical on the traded-only sample; it loses to Black-Scholes in Q3 2020, when the
index-vol correlation was about zero; LightGBM is better in calm years and worse and unstable in 2020.
Traded-only sample: 1,941,559 pairs (both legs of a pair must have traded).

## Neural networks run (2026-11-21)

Notebook 03 executed on the main sample: three networks x five seeds, 3x128 sigmoid, Adam 1e-4,
early stopping on 2019. Tables in `tables/main/neural/`, figures `figures/main/neural_*.png`,
findings at the end of the notebook. Gain ratio vs Black-Scholes, calls / puts, seed mean (sd):

| | 2019 | 2020 |
|---|---|---|
| Hull-White | 0.042 / 0.101 | 0.191 / 0.161 |
| LightGBM | 0.076 / 0.122 | 0.156 / 0.067 |
| nn_direct (paper) | 0.087 (0.015) / 0.154 (0.006) | -0.076 (0.079) / -0.840 (0.298) |
| nn_resid (paper) | 0.095 (0.002) / 0.137 (0.022) | -0.105 (0.065) / -0.138 (0.038) |
| nn_resid_hw (T, delta; output x vega/(S sqrt T)) | 0.071 (0.003) / 0.146 (0.005) | 0.167 (0.008) / 0.134 (0.009) |

- The paper's result replicates in the calm year: both networks beat Hull-White on 2019.
- Both of the paper's networks are worse than plain Black-Scholes in 2020, in nearly every seed.
  Inputs leave the training range (theta 2.2 sd, iv 1.4 sd from the training mean) and the
  residual network applies a correction of the wrong sign.
- nn_resid_hw is stable and positive in 2020 but below Hull-White in every seed.
  **It was designed after seeing the 2020 failures** (`screening_single_seed.csv`), so its 2020
  number is not a clean out-of-sample test. Say so in the report.
- Validation gain is nearly identical across seeds while 2020 gain is not: a calm validation
  year cannot select for crisis robustness.
- Network predictions (seed averages) are saved in `data/processed/predictions/` for notebook 05.

Not done: networks on the traded-only sample, GBM tuning, GPU support in `NeuralHedger`.

## Transaction costs run (2026-11-21)

Notebook 05 executed on the main sample (2019 and 2020, all five models from saved predictions).
`src/costs.py` rewritten: vectorised, hedging episodes = runs of consecutive trading days (a gap
starts a new episode), set-up and unwind both charged. Tables in `tables/main/costs/`, figures
`figures/main/costs_*.png`.

- Rebalancing per day is 0.85-1.06 times Black-Scholes for every model, so costs are nearly
  identical across models (at 1 bp: ~0.43 pts per episode in 2019, ~0.56 in 2020; differences
  between models <= 0.03 pts against risk reductions of 0.5-9.4 pts).
- Net-of-cost gain ratios are flat from 0 to 5 bp; no ranking changes. Costs do not alter any
  conclusion of notebooks 02-03.
- Unexpected: scored on episode totals instead of one day, every model beats Black-Scholes in
  2020, including the paper's networks (nn_direct 0.19 / 0.09, nn_resid 0.11 / 0.14), and LightGBM
  (0.32 / 0.32) is ahead of Hull-White (0.19 / 0.28). Likely because a steady level error in the
  hedge ratio cancels over up-and-down days when summed but not when squared daily. NOT tested.
  Episodes overlap and share one index path, so this is weak evidence; report both horizons.
- Left out: option bid-ask, financing, market impact, intraday rebalancing.

## Heston known-model check run (2026-11-21)

Notebook 04 executed. `src/heston.py`: pricer integration range now adapts to maturity and
variance (the fixed range was off by up to 0.5 index points for 14-day options in a calm market),
`mv_delta` gives the exact minimum-variance delta dC/dS + (rho xi / S) dC/dv by central
differences, `simulate_pairs` returns a pairs table with the truth attached. Simulated table is
cached in `data/simulated/` (git-ignored, about 6 minutes to rebuild). Tables in `tables/heston/`,
figures in `figures/heston/`. Parameters: kappa 2, theta 0.04, xi 0.3, rho -0.7.

Gain vs Black-Scholes (networks: 5-seed average), with the true MV delta as the ceiling:

| test market | true | hw | gbm | nn_direct | nn_resid | nn_resid_hw |
|---|---|---|---|---|---|---|
| same parameters | 0.142 | 0.127 | 0.131 | 0.113 | 0.131 | 0.139 |
| high vol (v0 = 0.25) | 0.078 | 0.058 | 0.064 | -0.750 | 0.085 | -0.024 |
| low correlation (rho = -0.1) | 0.005 | -0.055 | -0.056 | -0.153 | -0.126 | -0.121 |

- Same market: every model recovers 80-98% of the achievable gain. The check passes.
- High vol, same law: the direct network collapses (as on real 2020); the residual network
  matches the truth; nn_resid_hw fails because Heston's correction scales with vega/(S sigma),
  not vega/(S sqrt T), and it has no vol input.
- Low correlation: every fitted model loses, as Hull-White did in Q3 2020. Inputs look normal.
- Limits: one parameter set, one path per test market, no theta input, lr 1e-3.
