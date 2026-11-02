# MF703Project: Machine Learning and Black-Scholes Delta Hedging

Replication and extension of Qiao & Wan (2024), *Enhancing Black-Scholes Delta Hedging via Deep Learning*
([arXiv:2407.19367](https://arxiv.org/abs/2407.19367)), on S&P 500 index option chains 2016–2020.

We compare four hedging strategies on their one-day hedging error:

| Model | What it does | Module |
|---|---|---|
| `bs` | Black-Scholes delta from implied vol (benchmark) | `models.BlackScholes` |
| `hw` | Hull & White (2017) quadratic regression correction | `models.HullWhite` |
| `gbm` | LightGBM on the residual (non-neural ML baseline) | `models.GBMResidual` |
| `nn_direct` | Feed-forward net learning the hedge ratio directly | `models.NeuralHedger(residual=False)` |
| `nn_resid` | Feed-forward net learning the correction to delta (the paper's method) | `models.NeuralHedger(residual=True)` |

Extensions beyond the paper: a Heston known-model check, transaction costs, the 2020 crisis window, and interpretability surfaces.

## Pipeline

```
data/raw/<year>/*.txt ──► src/build_dataset.py ──► data/processed/pairs.parquet
                          stage 1: parse, calendar-check, hard input checks, filter, validate  (src/load.py, src/checks.py, src/schema.py)
                          stage 2: pair each contract with itself h trading days later on the NYSE calendar; target, buckets, split
                                   │
                  src/models.py  (fit / predict)  ◄── one table, one interface
                                   │
                  src/evaluate.py (gain ratios by split, year, delta bucket, maturity)
                                   │
                  src/costs.py  ·  src/heston.py   (extensions)
```

One command builds the dataset from a config file (`configs/`); a copy of the settings is saved next to the output. Details, data-quality findings and per-year counts: `docs/data_pipeline.md`.

## Setup

```bash
conda create -n mf703 python=3.11 -y && conda activate mf703
pip install -r requirements.txt
```

## Using pip venv
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


Put the raw vendor files in `data/raw/<year>/` (see `data/README.md`; they are git-ignored, ~25 MB per month).

## Run

```bash
# Stages 1 + 2 from the project config: 2016-2020, one-day horizon, train 2016-18 / val 2019 / test 2020
python -m src.build_dataset --config configs/main_2016_2020.yaml

# variants (see configs/README.md)
python -m src.build_dataset --config configs/traded_only_2016_2020.yaml
python -m src.build_dataset --years 2016 --out data/processed      # one year, flags override the config
```

Then open the notebooks in order:

| Notebook | Owner | Contents |
|---|---|---|
| `01_data.ipynb` | data | runs the pipeline from the config, data-quality report, benchmark error by year |
| `02_baselines.ipynb` | baselines | Black-Scholes vs Hull-White vs GBM, gain-ratio tables and plots |
| `03_neural.ipynb` | ML | direct and residual networks, training curves, 2019/2020 results |
| `04_heston.ipynb` | simulation | Heston simulation, known-truth correction vs learned correction |
| `05_costs.ipynb` | simulation | multi-day P&L with transaction costs, cost sweep |

## Conventions

- Option price = bid-ask midpoint. Hedging error = `dV - delta * dS` in index points per option.
- Filters (defaults in `load.FilterConfig`, set per config): DTE ≥ 14, 0.05 ≤ |delta| ≤ 0.95, bid > 0, bid ≤ ask. `require_volume: true` / `--traded-only` gives the volume > 0 subsample.
- Pairing uses the NYSE calendar (`pandas_market_calendars`): a day missing from the vendor files drops a pair, never bridges one.
- Splits are chronological and set in the config (default `proposal`: train 2016–18, val 2019, test 2020; also `paper` and `modern` in `build_dataset.SPLIT_SETS`). Never random-split: the same contract appears on consecutive days.
- Every model exposes `fit(pairs)` / `predict(pairs)` and writes a `delta_<name>` column; `evaluate.gain_ratio` reads those columns.
- Figures go to `figures/`, tables to `tables/`, with stable filenames so the report can reference them.

## Tests

```bash
pytest tests/
```

Twelve tests on synthetic vendor files: parsing, filters, the units and schema checks, calendar handling (gaps drop pairs, holiday rows are removed), pairing arithmetic, an end-to-end build from a config, Hull-White recovery and the gain ratio. GitHub Actions runs them on every pull request.

## References

- Hull, J., White, A. (2017). Optimal delta hedging for options. *Journal of Banking & Finance* 82, 180–190.
- Qiao, C., Wan, X. (2024). Enhancing Black-Scholes Delta Hedging via Deep Learning. arXiv:2407.19367.
- Heston, S. (1993). A closed-form solution for options with stochastic volatility. *RFS* 6, 327–343.
