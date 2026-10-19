# INFORMS 2026: EOD utility-variation study

Replays a constrained Cobb-Douglas portfolio strategy on end-of-day S&P 500 data.
Package `VLINFORMS2026` lives in `code/`.

## Run order

1. `julia scripts/00_fetch_data.jl`: copy the price/sector files into `data/` and download FRED DTB3.
2. Test: `julia --project=code code/test/runtests.jl`.

Later scripts are added in numeric order.
