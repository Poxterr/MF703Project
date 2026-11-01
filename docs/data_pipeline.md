# Data pipeline

This note documents stage 1 of the project: how raw S&P 500 option chains become the table every
hedging model trains on. It is written to be lifted into the Data section of the final report, and
to let anyone rebuild the dataset, or extend it to new years or another vendor, without asking.

## Source

End-of-day SPX option chains from [OptionsDX](https://www.optionsdx.com/shop/) (product "SPX
Option Chains", end-of-day variant), one text file per month named `spx_eod_YYYYMM.txt`, about
25 MB each. Each row is one (quote date, expiry, strike) with the call and the put side by side:
bid, ask, last, volume, quote sizes, implied volatility and the Greeks (delta, gamma, vega, theta,
rho), all stamped 16:00 Eastern. The index level is included on every row.

The project sample is **2016–2020** (60 files, 1.9 GB raw). Files for 2010–2015 and 2021–2023 have
also been downloaded and run through the same pipeline for the replication and extension configs.
The raw files are not in git (size, and the vendor's terms); they are shared through the group's
Drive folder and placed under `data/raw/<year>/`.

## What the pipeline does

```
data/raw/<year>/spx_eod_YYYYMM.txt
        │  stage 1  (src/load.py + src/build_dataset.clean_file)
        │   parse vendor text ─► one row per option (call / put) ─► exchange-calendar check
        │   ─► hard input checks ─► sample filters ─► schema assertion ─► Parquet + quality report
        ▼
data/processed/spx_eod_YYYYMM.parquet     load_report.csv     data_quality.md
        │  stage 2  (src/build_dataset.build_pairs_year)
        │   pair each contract with itself h trading days later on the NYSE calendar
        │   ─► dV, dS, y = dV/dS, w = dS², benchmark error ─► buckets, sentiment features, split
        ▼
data/processed/pairs_h1_parts/year=YYYY.parquet  ─► merge ─►  pairs.parquet  (+ summary, meta, build_config.yaml)
```

One command builds everything from a config file:

```bash
python -m src.build_dataset --config configs/main_2016_2020.yaml
```

The config pins years, horizon, filters, splits and vendor; a copy is written next to the output
as `build_config.yaml`, so every `pairs.parquet` records how it was made. Processing a new year is
`--years 2021` (or a new config); another vendor's files need one entry in `load.COLUMN_MAPS` and
`load.VENDOR_UNITS`.

### Stage 1: clean

1. **Parse.** Bracketed headers are stripped, blank fields become NaN, `"12 x 365"` quote sizes are
   split into bid and ask size. Numeric columns are cast to float explicitly so the output dtype
   never depends on the input.
2. **Reshape.** The wide call/put row becomes two rows with a `type` column and one standard set
   of column names (`src/schema.py`, `OPTION_DAY_SCHEMA`, 21 columns).
3. **Units.** The vendor quotes vega per 1 percentage point of volatility; the loader multiplies
   by 100 so all Greeks are in the Black–Scholes convention used by Hull and White (2017).
4. **Exchange calendar.** Rows dated on days the NYSE was closed are dropped and recorded.
5. **Hard input checks** (`src/checks.py`) stop the run on input that would corrupt every model
   silently: a file without the vendor's columns (wrong vendor), quote dates outside the month in
   the file name, a month with no trading days left, implied vol not in decimal, and Greeks that
   disagree with Black–Scholes recomputed from the vendor's own implied vol (catches the vega
   units problem, a swapped delta column, or IV in percent).
6. **Sample filters**, following Hull and White (2017) and Qiao and Wan (2024): at least 14
   calendar days to expiry, $0.05 \le |\delta| \le 0.95$, bid $> 0$, bid $\le$ ask. The traded-only
   variant (`require_volume: true`) additionally requires volume $> 0$ on the quote day.
7. **Schema assertion** against `OPTION_DAY_SCHEMA`, then one Parquet per raw file and one row in
   `load_report.csv` (counts, missing days, zero-bid days, units-check results).

### Stage 2: pair

For option $i$ quoted on trading day $t$, the pipeline finds the same contract (type, expiry,
strike) exactly $h$ trading days later **on the NYSE calendar**, never "the next date that happens to
be in the file". A missing day in the vendor data therefore drops a pair instead of silently
turning a one-day hedge into a two-day move. With mid prices $V$ and index level $S$:

$$
\Delta V = V_{t+h} - V_t, \qquad
\Delta S = S_{t+h} - S_t, \qquad
y = \frac{\Delta V}{\Delta S}, \qquad
w = \Delta S^2 .
$$

The one-step hedging error of a hedge ratio $\delta$ is $\varepsilon = \Delta V - \delta\,\Delta S$,
and since $\varepsilon^2 = w\,(y-\delta)^2$, every model is a weighted regression of $y$ (or of
$y - \delta_{BS}$ for residual models) with weights $w$. The table also carries the Black–Scholes
benchmark error `err_bs`, the change in implied vol `d_iv`, the paper's signed delta bucket and
maturity class, the index log-return (and VIX when a FRED file is supplied), and a chronological
`split` column so no notebook can choose its own split.

Pairs are built one year at a time and streamed into `pairs.parquet`, so memory stays small at
any scale (the 2010–2023 run is 13 million rows).

## The 2016–2020 dataset

Built with `configs/main_2016_2020.yaml` (all quoted options, one-day horizon, train 2016–18,
validate 2019, test 2020). 60 files, 82 seconds on a laptop-class machine.

| Year | Trading days in data | Raw strike rows | Option-days after filters | Pairs | Calls / puts | Start days lost to gaps | $\mathbb{E}[\varepsilon_{BS}^2]$ calls | puts | mean $|\Delta S|$ |
|---|---|---|---|---|---|---|---|---|---|
| 2016 | 251 | 940,596 | 661,539 | 610,126 | 290,087 / 320,039 | 1 | 2.55 | 2.17 | 12.0 |
| 2017 | 250 | 1,023,605 | 736,105 | 667,606 | 352,248 / 315,358 | 1 | 1.79 | 1.31 | 7.2 |
| 2018 | 251 | 1,382,559 | 1,328,518 | 1,224,338 | 714,586 / 509,752 | 0 | 10.24 | 10.30 | 20.4 |
| 2019 | 247 | 1,452,262 | 1,367,654 | 1,236,295 | 719,069 / 517,226 | 5 | 4.58 | 5.93 | 16.2 |
| 2020 | 250 | 1,736,777 | 1,845,164 | 1,681,366 | 864,630 / 816,736 | 4 | 82.28 | 78.66 | 42.7 |
| **Total** | 1,249 | 6,535,799 | 5,938,980 | **5,419,731** | 2,940,620 / 2,479,111 | 11 | | | |

Benchmark errors are in index points per option. For comparison, Qiao and Wan (2024) had about
2.07 million quotes after the same filters over 2010–2019; the OptionsDX files include weeklies and
a denser strike grid, so our five years give more than twice their ten.

### Data quality findings

- **No duplicate (date, expiry, strike) rows, two crossed quotes in 6.5 million, one index level
  per day** in every file.
- **Ten NYSE trading days are absent** from the vendor files: 2016-04-18, 2017-09-27, 2019-03-01,
  2019-04-17, 2019-05-24, 2019-06-07, 2019-09-05, 2020-06-18, 2020-11-27, 2020-12-24. Each costs
  one start day of pairs (the day before it); nothing is bridged.
- **One non-trading day with rows:** 2018-12-05 (the market closed for President Bush's funeral;
  the vendor still produced 12,036 rows). Dropped.
- **Zero bids** on 6–9% of raw option rows per year, almost all deep out-of-the-money; removed by
  the bid $> 0$ filter.
- **Units check:** vendor delta agrees with Black–Scholes delta from the vendor's IV to a median
  0.005–0.012; rescaled vega is within 1% of the formula, every year.
- **Zero volume** on roughly two thirds of quoted options. The main dataset keeps them (SPX
  market-maker quotes are tight, median spread 2.3% of mid); `traded_only_2016_2020.yaml` builds the
  paper-style subsample for a robustness check. With all quotes kept, calls pile up in the 0.9
  delta bucket (deep in-the-money, quoted but never traded); the traded-only sample does not.
- **2020 dominates any pooled statistic.** Its benchmark error is 16–35 times the other years'
  and the mean daily index move is 2–6 times larger. Results must be reported by year and by the
  February–April 2020 window separately, which is what the proposal's crisis extension does.

## Reproducing and extending

```bash
pip install -r requirements.txt
# place spx_eod_YYYYMM.txt under data/raw/<year>/
python -m src.build_dataset --config configs/main_2016_2020.yaml     # the project dataset
python -m src.build_dataset --config configs/traded_only_2016_2020.yaml
python -m src.build_dataset --years 2021 --out data/processed          # add a year to an existing build
pytest tests/                                                          # 12 tests, a few seconds
```

Tests run on synthetic vendor files and cover parsing, filters, the units and schema checks, the
calendar logic (a missing day drops a pair; a holiday row is removed), the pairing arithmetic, and
one end-to-end build from a config file. GitHub Actions runs them on every pull request.
