# INFORMS 2026: EOD utility-variation study

Replays a constrained Cobb-Douglas portfolio strategy on end-of-day S&P 500 data, and runs the screened
utility-panel study on top of it. Package `VLINFORMS2026` lives in `code/`. All performance numbers are
growth rates g = ln(S_end/S_start)/(k·Δt) in 1/yr with Δt = 1/252.

## Run order

`00 → 10 → 11 → 11 --fixed → 12 → 13`

1. `julia scripts/00_fetch_data.jl`: copy the price/sector files into `data/` and download FRED DTB3.
2. `julia scripts/10_calibrate.jl`: calibrate on 2014-01-03 → 2024-12-31. Writes `results/calibration.jld2` and
   `results/calibration_summary.toml` (λ, σ_m, annualized vol cap, m_m, v_m, ξ gain G, universe and sector counts).
3. `julia -t auto scripts/11_run_study.jl`: Tier 1 grid (allocators × screens × hedge × seeds). Per-run files in
   `results/runs/` (not committed; reruns skip finished runs), summary in `results/tier1_summary.csv`.
   `--smoke` runs 2 runs (CD × γ screen × hedge on/off) into `results/smoke/`.
4. `julia -t auto scripts/11_run_study.jl --fixed`: fixed-schedule grid (no re-pick on band exit), `results/fixed_summary.csv`.
5. `julia scripts/12_tables.jl [dir] [--smoke]`: comparison tables to `<dir>/tables/*.csv` (default `results/`),
   also printed as Markdown. Smoke check: `julia scripts/12_tables.jl results/smoke --smoke`.
6. `scripts/13_figures.jl`: figures (uses `scripts/Project.toml`).

Tests: `julia --project=code code/test/runtests.jl`. Settings live in `config/study.toml`.
`data/`, `results/runs/` and `results/smoke/` are never committed; summaries, tables, figures and
`results/calibration*` are.

## Plan decisions

Choices the plan makes where the spec is open (marked **(plan decision)** in the docstrings):

- Growth rates only: bandit reward, Sharpe, alpha and every table column use g.
- Calibration window 2014-01-03 → 2024-12-31; run window 2025-01-02 → 2026-09-04; no look-ahead (orders fill at the next open).
- λ (residual shrinkage) and the volatility cap (SPY growth volatility, annualized) are calibrated on the training window.
- The deterministic γ screen runs one seed; bandit and random screens run `n_seeds` seeds (base seed 5660).
- An infeasible allocation keeps previous holdings; the hedge pool and β cap are described in the `HedgePool.jl` and `StudyBacktest.jl` docstrings.
- Every window, lag, limit and cost comes from `config/study.toml`.
