# Data folder

```
data/
  raw/                      vendor files exactly as downloaded, one folder per year
    2016/spx_eod_201601.txt ... spx_eod_201612.txt
    2017/
    2018/
    2019/
    2020/
  processed/                written by the pipeline, never edited by hand
    spx_eod_YYYYMM.parquet  stage 1 output, one per raw file
    load_report.csv         quality report, one row per raw file
    pairs.parquet           stage 2 output: the table every model trains on
```

## Getting the raw files

The raw files are **not in git** (about 25 MB per month, 1.5 GB for five years; GitHub rejects
files over 100 MB and the repository would be unusable). They live in the group's shared Drive
folder. Download the year folders into `data/raw/` keeping the names `spx_eod_YYYYMM.txt`.

## Building the processed data

```bash
# all years present under data/raw/
python -m src.load --out data/processed

# one year only
python -m src.load --years 2016 --out data/processed

# then the pairs table
python -m src.pairs data/processed --out data/processed/pairs.parquet
```

Stage 1 takes about 10 seconds per year; stage 2 a few seconds. `processed/` is also git-ignored;
rebuild it from `raw/` on each machine, or copy `pairs.parquet` (about 40 MB per year) from Drive.

## Checking a new year

After adding a year, look at `processed/load_report.csv`: every file should show
`dup_keys = 0`, `crossed_quotes` of at most a handful, `underlying_consistent = True`, and a
`median_rows_per_day` in the low thousands. March 2020 will show a lower `min_rows_per_day`
than other months; that is the crisis, not a bad file.
