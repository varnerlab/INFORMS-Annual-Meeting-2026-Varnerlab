# INFORMS 2026: EOD utility-variation study

Replays a constrained Cobb-Douglas portfolio strategy on end-of-day S&P 500 data, and runs the screened
utility-panel study on top of it. Package `VLINFORMS2026` lives in `code/`. All performance numbers are
growth rates g = ln(S_end/S_start)/(k·Δt) in 1/yr with Δt = 1/252.

## Run order

`00 → 10 → 11 → 11 --fixed → 15 → 11 --universe → 12 → 13`

1. `julia scripts/00_fetch_data.jl`: copy the price/sector files into `data/` and download FRED DTB3.
2. `julia scripts/10_calibrate.jl`: calibrate on 2014-01-03 → 2024-12-31. Writes `results/calibration.jld2` and
   `results/calibration_summary.toml` (λ, σ_m, annualized vol cap, m_m, v_m, ξ gain G, universe and sector counts).
3. `julia -t auto scripts/11_run_study.jl`: Tier 1 grid (allocators × screens × hedge × seeds). Per-run files in
   `results/runs/` (not committed; reruns skip finished runs), summary in `results/tier1_summary.csv`.
   `--smoke` runs 2 runs (CD × γ screen × hedge on/off) into `results/smoke/`.
4. `julia -t auto scripts/11_run_study.jl --fixed`: fixed-schedule grid (no re-pick on band exit), `results/fixed_summary.csv`.
5. `julia scripts/15_regression_check.jl`: compares the rerun `results/tier1_summary.csv` and
   `results/fixed_summary.csv` with the versions committed at `HEAD`, joined on `label`, over every
   growth-scorecard metric and run statistic. Exits 1 unless every committed run is present and every value
   is exactly equal. Run it after a code change and before committing new summaries.
6. `julia -t auto scripts/11_run_study.jl --universe`: the ticker-picker addendum's 3,804 runs (universe-wide
   screens and random controls at K ∈ {5, 8, 13, 21, 34}, plus the user's 13- and 15-name lists), summary in
   `results/universe_summary.csv`.
7. `julia scripts/12_tables.jl [dir] [--smoke]`: comparison tables to `<dir>/tables/*.csv` (default `results/`),
   also printed as Markdown. Smoke check: `julia scripts/12_tables.jl results/smoke --smoke`. With
   `universe_summary.csv` present it also writes `size_sweep.csv`, `matched_percentile.csv`,
   `head_to_head_K13.csv` and `quota_cost.csv`.
8. `julia --project=scripts scripts/13_figures.jl`: slide figures (1600×900 PNG) to `results/figures/`, from
   `results/*_summary.csv`, `results/runs/`, `config/study.toml` and `results/calibration_summary.toml`
   (figs 6–7, the size sweep, need `universe_summary.csv`).
   It uses its own environment (`scripts/Project.toml`, Plots.jl with VLINFORMS2026 developed from `code/`);
   once per checkout, run `julia --project=scripts -e 'using Pkg; Pkg.instantiate()'`.

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
- ξ's short and long averages and g̃_M are exponential moving averages (EMA, as in CHEME-5660 L8b) where the
  spec says "moving average": ξ uses spans `xi_short`/`xi_long` (21/63 days) and g̃_M a span of `gtilde_window`
  (252 days).
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

Ticker-picker addendum (spec `docs/superpowers/specs/2026-10-20-ticker-picker-addendum-design.md`), "my call"
items the code implements:

- The liquid shortlist ranks names by the median of close × volume (USD/day) over the trailing 63 trading days.
- The liquid shortlist keeps the top 100 names by that dollar volume.
- The correlation-aware γ screen accepts a name only if its 252-day growth-rate correlation with every name already accepted is ≤ 0.7, fixed before any run.
- A screen pick's per-stock cap is max(w_max, 1.5/K), so a K-name basket can be fully invested with room to tilt.
- A name that is both a screen pick and a hedge-pool name gets the pick's cap; pool-only names keep w_max.
- The 130/30 shorts are unchanged under every screen (bottom 3 γ < 0 per sector, excluding the pool), and short names keep w_max.
- The new screens get no fixed-schedule variant; the base fixed-schedule grid already answers that question.

The universe-wide screens, their thresholds and the K grid were designed after the Tier 1 results were known,
so their comparison is exploratory, not out-of-sample. The user's 13-name list (AAPL, MSFT, INTC, MU, AMD, NVDA,
GS, BAC, WFC, C, F, GM, JNJ) and the 15-name sensitivity list (the same plus WMT and PG) were chosen in advance,
without 2025 data.

Throughout, performance is measured by growth rates g = ln(S_end/S_start)/(k·Δt) in 1/yr, with no simple or log returns.
