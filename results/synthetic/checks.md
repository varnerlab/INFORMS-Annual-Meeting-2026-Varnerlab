# Synthetic-market checks V1–V9

Written by `scripts/24_synthetic_checks.jl` on 2026-10-27 (wall time 35.5 s on 30 threads).

- Check paths: the first 50 paths by `typical_rank` in `paths.csv` (ids 2594, 4726, 8301, 3884, 1811, 5068, 1799, 8958, 7053, 3716, 4977, 2730, 935, 1781, 5058, 709, 9269, 2849, 861, 264, 7058, 9202, 5512, 2341, 1121, 7949, 4025, 8304, 9530, 2837, 6053, 1373, 3770, 6765, 924, 1018, 2621, 6226, 9621, 3901, 3449, 7315, 91, 5026, 7453, 4268, 7946, 2273, 6221, 7127).
- Each check path is recalibrated on 2014-01-03 → 2024-12-31 with the study's `calibrate`, then compared with the real market over the same window.
- V1 uses all 10000 paths in `paths.csv`.
- A number with one value per path is the median over the 50 paths. A number with one value per name is each name's median over the paths, then the median over the 373 names.
- `pass` is `missing` for a row that is reported but not gated.
- V8 is the study's full test suite, run on its own: `julia --project=code code/test/runtests.jl`.
- Median share of name-days on which the composition scaled β and the sector loading down (clipped): 0.0883.

**19 of 22 gates pass. Failing: V5 (mean residual correlation within Energy); V5 (mean residual correlation within Utilities); V9 (cross-name SD of OLS α, 2014–2024 (yr⁻¹)).**

| check | statistic | synthetic | reference | threshold | pass |
|---|---|---|---|---|---|
| V1 | SPY growth 2014–2024, median over 10000 paths (yr⁻¹) | 0.1054 | 0.1061 | within 0.02 of reference | true |
| V2 | equal-weight growth, rebalanced daily minus bought and held (yr⁻¹) | -0.01999 | -0.02016 | within 0.02 of reference | true |
| V3 | variance ÷ variance of the name's own draw | 0.9987 | 1.0 | in [0.95, 1.05] | true |
| V3 | absolute gap between the OLS β and the mean composed β | 0.02172 | 0.0 | ≤ 0.05 | true |
| V3 | variance ÷ real variance | 1.068 | 1.0 | report only | missing |
| V4 | abs-growth autocorrelation at lag 1, synthetic − own draw | -0.03895 | 0.0 | within 0.05 of reference | true |
| V4 | abs-growth autocorrelation at lag 5, synthetic − own draw | -0.002703 | 0.0 | within 0.05 of reference | true |
| V4 | abs-growth autocorrelation at lag 20, synthetic − own draw | -0.0009381 | 0.0 | within 0.05 of reference | true |
| V4 | abs-growth autocorrelation at lag 1, synthetic − real | -0.09682 | 0.0 | report only | missing |
| V4 | abs-growth autocorrelation at lag 5, synthetic − real | -0.1767 | 0.0 | report only | missing |
| V4 | abs-growth autocorrelation at lag 20, synthetic − real | -0.1054 | 0.0 | report only | missing |
| V5 | mean residual correlation within Communication Services | 0.05036 | 0.0937 | within 0.05 of reference | true |
| V5 | mean residual correlation within Consumer Discretionary | 0.09196 | 0.1122 | within 0.05 of reference | true |
| V5 | mean residual correlation within Consumer Staples | 0.1728 | 0.204 | within 0.05 of reference | true |
| V5 | mean residual correlation within Energy | 0.478 | 0.5416 | within 0.05 of reference | false |
| V5 | mean residual correlation within Financials | 0.204 | 0.2386 | within 0.05 of reference | true |
| V5 | mean residual correlation within Health Care | 0.1022 | 0.1279 | within 0.05 of reference | true |
| V5 | mean residual correlation within Industrials | 0.1177 | 0.1354 | within 0.05 of reference | true |
| V5 | mean residual correlation within Information Technology | 0.08346 | 0.102 | within 0.05 of reference | true |
| V5 | mean residual correlation within Materials | 0.1194 | 0.166 | within 0.05 of reference | true |
| V5 | mean residual correlation within Real Estate | 0.3169 | 0.366 | within 0.05 of reference | true |
| V5 | mean residual correlation within Utilities | 0.4759 | 0.5493 | within 0.05 of reference | false |
| V5 | mean residual correlation across sectors | 8.159e-5 | 0.03094 | report only | missing |
| V6 | SD of the online β̂ ÷ the real SD | 0.987 | 1.0 | in [0.9, 1.1] | true |
| V6 | lag-252 autocorrelation of the online β̂ | 0.1863 | 0.2062 | report only | missing |
| V6 | share of paths with a name whose β̂ ≤ 0 in 2025–26 (reference: real count of names) | 1.0 | 120.0 | report only | missing |
| V9 | cross-name SD of OLS α, 2014–2024 (yr⁻¹) | 0.0894 | 0.0764 | within 0.01 of reference | false |
| V9 | correlation of OLS α, 2014–2019 vs 2020–2024 | 0.1266 | 0.1748 | within 0.05 of reference | true |
| V7 | market-data fields that differ when path 2594 is built twice | 0.0 | 0.0 | = 0 | true |
