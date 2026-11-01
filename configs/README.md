# Build configs

One YAML per dataset. `python -m src.build_dataset --config configs/<name>.yaml` builds it;
a copy of the settings is written to the output folder as `build_config.yaml`.

| Config | Years | Horizon | Sample | Split (train / val / test) | Output |
|---|---|---|---|---|---|
| `main_2016_2020.yaml` | 2016-2020 | 1 day | all quoted (bid > 0) | 2016-18 / 2019 / 2020 | `data/processed/` |
| `traded_only_2016_2020.yaml` | 2016-2020 | 1 day | volume > 0 | same | `data/processed_traded_only/` |
| `weekly_2016_2020.yaml` | 2016-2020 | 5 days | all quoted | same | `data/processed/pairs_h5.parquet` |
| `paper_2010_2019.yaml` | 2010-2019 | 1 day | volume > 0 | 2010-17 / 2018 / 2019 | `data/processed_paper/` |

To process a new year or another vendor's files: add the raw files under `data/raw/<year>/`,
copy a config, edit `years` (and `vendor`, after adding its column map in `src/load.py`), run.
Any command-line flag overrides the file, e.g. `--years 2021` or `--traded-only`.
