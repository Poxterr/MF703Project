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
