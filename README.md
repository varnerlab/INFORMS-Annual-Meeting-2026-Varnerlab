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

Choices the plan or the build makes where the spec is open (marked **(plan decision)** in the docstrings).

Plan:

- Config field names are ASCII, so `config/study.toml` keys equal `MyStudySettings` field names.
- The hedge-pool tie-break: a hedge name holds more than its floor only when a binding limit (β cap or vol cap) makes it worth its budget.
- "Share of days the β cap binds" is counted over decision days (`cap_bind_share`), the only days the cap is enforced.
- The figures use Plots.jl in a separate scripts environment.
- The old 10-19 `run_backtest` stays; `run_study` reuses its fill and cost helpers.

Spec "my call" items the code implements:

- The ξ gain G is L8b's calibrated gain (|ξ| ≤ 1 on 99% of 2014–2024 days), not the trial's fixed 20.
- The vol cap is SPY's 2014–2024 growth-rate SD (annualized ≈ 17%).
- γ uses the signed numerator with |β| floored at β_floor = 0.1 in the denominator.
- The online β prior uses SPY's 2014–2024 mean and variance (CHEME-5660 `ewls_init`).
- The β cap is β_max = max(β_min, β_ref − κ_β·ξ), which is max(0, 1 − ξ) by default.
- The hedge pool holds up to 5 names whose online β and down-day β are both ≤ 0, most negative down-day β first, and pool names are never shorted.
- Hedge-off control: every run has a paired twin without the cap or pool.

Rulings made during the build, each of which changes behavior:

- Hedge names are minimized in a second, lexicographic solve (utility names fixed), because a 1e-6 tie-break alone is below the solver's accuracy.
- CES with η < 1 uses a rescaled power cone (same optimum, accurate at large equity).
- Stone–Geary with no γ > 0 still meets the β cap (buys the least hedge, or keeps holdings with `:infeasible`).
- On an `:infeasible` or `:solver_failed` decision, `run_study` places no orders and keeps every holding, and the band and stats describe the kept book.
- After a decision that leaves the book all cash, no band is set, so only cash revisit, horizon or drawdown can trigger the next decision (as in the trial).
- An empty 130/30 problem records `:no_preferred` and goes to cash instead of erroring.
- A study context that fails to build writes error files for its runs, and the sweep continues.

Throughout, performance is measured by growth rates g = ln(S_end/S_start)/(k·Δt) in 1/yr, with no simple or log returns.
