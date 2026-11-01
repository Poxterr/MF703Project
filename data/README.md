# Data folder

```
data/
  raw/                      vendor files exactly as downloaded, one folder per year (git-ignored)
    2016/spx_eod_201601.txt ... spx_eod_201612.txt
    2017/  2018/  2019/  2020/           the project sample
    2010-2015/, 2021-2023/               optional: replication and extension configs
  processed/                written by the pipeline, never edited by hand
    spx_eod_YYYYMM.parquet  stage 1 output, one per raw file              (git-ignored)
    pairs_h1_parts/         stage 2 output, one Parquet per year           (git-ignored)
    pairs.parquet           all pairs in one file: the modelling table    (git-ignored, ~335 MB for 2016-2020)
    load_report.csv         one quality row per raw file                 (committed)
    data_quality.md         every missing / dropped day, human-readable  (committed)
    pairs_h1_summary.csv    counts and benchmark error by year           (committed)
    build_config.yaml       the exact settings that produced this build  (committed)
```

## Getting the raw files

Source: [OptionsDX](https://www.optionsdx.com/shop/), product "SPX Option Chains", end-of-day
variant. The files are monthly text files named `spx_eod_YYYYMM.txt`, about 25 MB each (1.9 GB
for 2016-2020). They are **not in git**: GitHub rejects files over 100 MB and the vendor's terms
do not allow redistribution. The group shares them through the Drive folder; download the year
folders into `data/raw/` keeping the names unchanged.

## Building the processed data

```bash
python -m src.build_dataset --config configs/main_2016_2020.yaml   # the project dataset (~80 s)
python -m src.build_dataset --years 2016 --out data/processed      # one year only
```

See `configs/README.md` for the other datasets (traded-only, weekly, paper replication) and
`docs/data_pipeline.md` for what the pipeline does and what it found in the data.

## Checking a new year

After adding a year, read `processed/data_quality.md`: it lists every trading day absent from the
files and every row dated on a closed-market day. In `load_report.csv`, every file should show
`dup_keys = 0`, `underlying_consistent = True`, a `vega_ratio_to_bs` near 1.0 and a
`delta_median_abs_diff` near 0.01. The pipeline stops by itself if a file is in the wrong layout,
misnamed, or has Greeks in the wrong units.
