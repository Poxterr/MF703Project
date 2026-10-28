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
