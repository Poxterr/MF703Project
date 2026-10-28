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
raw vendor files ──► src/load.py ──► data/processed/*.parquet ──► src/pairs.py ──► pairs.parquet
                        (clean, filter, validate)                   (next-day pairing, target, split)
                                                                             │
                   src/models.py  (fit / predict)  ◄────────────────────────┘
                        │
                   src/evaluate.py (gain ratios by split, year, delta bucket, maturity)
                        │
                   src/costs.py  ·  src/heston.py   (extensions)
```

Each stage reads the previous stage's file, so the four workstreams can run in parallel once `pairs.parquet` exists.

## Setup

```bash
conda create -n mf703 python=3.11 -y && conda activate mf703
pip install -r requirements.txt
```

Put the raw vendor files in `data/raw/<year>/` (see `data/README.md`; they are git-ignored, ~25 MB per month).

## Run

```bash
# Stage 1: clean every raw file into Parquet, with a quality report
python -m src.load --out data/processed          # all years under data/raw/
python -m src.load --years 2016 --out data/processed   # one year

# Stage 2: build the pairs table with train/val/test split
python -m src.pairs data/processed --out data/processed/pairs.parquet
```

Then open the notebooks in order:

| Notebook | Owner | Contents |
|---|---|---|
| `01_data.ipynb` | data | runs stages 1–2, data-quality checks, descriptive figures |
| `02_baselines.ipynb` | baselines | Black-Scholes vs Hull-White vs GBM, gain-ratio tables and plots |
| `03_neural.ipynb` | ML | direct and residual networks, training curves, 2019/2020 results |
| `04_heston.ipynb` | simulation | Heston simulation, known-truth correction vs learned correction |
| `05_costs.ipynb` | simulation | multi-day P&L with transaction costs, cost sweep |

## Conventions

- Option price = bid-ask midpoint. Hedging error = `dV - delta * dS` in index points per option.
- Filters (defaults in `load.FilterConfig`): DTE ≥ 14, 0.05 ≤ |delta| ≤ 0.95, bid > 0. `--traded-only` gives the volume > 0 subsample.
- Splits are chronological and set in `pairs.DEFAULT_SPLITS`: train 2016–18, val 2019, test 2020. Never random-split: the same contract appears on consecutive days.
- Every model exposes `fit(pairs)` / `predict(pairs)` and writes a `delta_<name>` column; `evaluate.gain_ratio` reads those columns.
- Figures go to `figures/`, tables to `tables/`, with stable filenames so the report can reference them.

## Tests

```bash
pytest tests/
```

The tests run on a tiny synthetic file and check the loader schema, the pairing arithmetic, and that Hull-White and the gain ratio reproduce known values.

## References

- Hull, J., White, A. (2017). Optimal delta hedging for options. *Journal of Banking & Finance* 82, 180–190.
- Qiao, C., Wan, X. (2024). Enhancing Black-Scholes Delta Hedging via Deep Learning. arXiv:2407.19367.
- Heston, S. (1993). A closed-form solution for options with stochastic volatility. *RFS* 6, 327–343.
