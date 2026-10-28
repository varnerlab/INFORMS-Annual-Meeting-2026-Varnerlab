# INFORMS 2026 Talk Slides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 8-slide INFORMS 2026 deck (plus three backup topics) as a XeLaTeX Beamer PDF whose every number comes
from committed results through one data script.

**Architecture:** Julia functions in `code/src/TalkData.jl` (tested in `code/test/test_talk_data.jl`) turn committed
results into small CSV and `.tex` files under `talk/data/`. `scripts/30_talk_data.jl` calls them and checks the
slide claims against the data. The deck in `talk/` uses an adapted copy of the CHEME 5660 `vnslides.sty`. It draws
every figure in TikZ/pgfplots from those files, so figure text uses the slide fonts. `talk/check_deck.sh` builds the
deck and lints it.

**Tech Stack:**
- Julia 1.12 with the `VLINFORMS2026` package (`code/`) and the `scripts/` environment (CSV, DataFrames, JLD2);
- XeLaTeX with `latexmk`, Beamer, TikZ, pgfplots and the `fillbetween` library, `pgfplotstable`, and `qrcode`;
- poppler (`pdfinfo`, `pdffonts`, `pdftoppm`).

**Spec:** `docs/superpowers/specs/2026-10-28-informs-talk-slides-design.md`

## Global Constraints

- **Deadline:** slides are due October 30. Main slides come before backups.
- **Template:** CHEME 5660 Beamer style (`vnslides.sty`), XeLaTeX, `\documentclass[aspectratio=169,11pt]{beamer}`.
- **Fonts:** Helvetica Neue Light, Medium and Bold.
- **Colors:**
  - ink #333333 (headings);
  - body #5E5E5E;
  - coral accent #FF644E (`\hilite`, the title rule, the SPY line);
  - link blue #579ACA.
- **Template furniture:** no disclaimer slide, no learning objectives and no lecture footer. Slide numbers are small,
  bottom right, in grey on every slide except the title. No other footer.
- **Forecast bands (L5a style):** 68/95/99% bands in `deepskyblue1` (#00BFFF) with transparency, the realized path in
  red (#FF0000) at line width 3, and a `gray95` (#F2F2F2) plot background.
- **Dot plots:**
  - rows are ticker pickers grouped by basket size, with small uppercase group labels in link blue;
  - one dot per utility;
  - hedge off is a filled #579ACA dot just above the row line;
  - hedge on is a hollow dot with a #2F6FAE outline just below it;
  - a pale #DBE7F3 bar spans each row's lowest to highest dot;
  - SPY buy-and-hold is a vertical coral dashed line, and the equal-weight universe is a grey dotted line;
  - the legend reads "hedge off", "hedge on", "SPY buy-and-hold", "equal-weight universe" and "one dot per utility".
- **Labels:** results are labeled "Real-data backtest, Jan 2025 – Sep 2026". Nothing presents backtest results as live
  trading.
- **Words:**
  - display names only, never internal labels (`global_gamma`, `l8b_13`, `fixed:`, "SCS share", run labels);
  - "returns" is never used for performance; ending wealth ÷ starting wealth is fine;
  - the hand-picked lists are never called "hindsight";
  - no em dashes (`---`) in slide copy, a 5660 house rule;
  - plain, specific language with no hype.
- **Numbers:** no pooled medians across utilities; results are picker × utility, one dot per utility, with SPY. Every
  plotted or quoted number comes from a committed results file through `scripts/30_talk_data.jl`.
- **Real-run values:** the median over seeds 5660–5689 for every picker × utility × hedge cell.
- **Conclusions:** the takeaway text is the user-approved text in the spec, with its numbers set by macros from
  `talk/data/numbers.tex`. The large-book sentence stays off the slide until the user approves it.
- **Runs:** start Julia with `OMP_NUM_THREADS=1 KMP_GTID_MODE=2`.
- **Git:** nothing is pushed, tagged, merged or made public. Commit on `feat/synthetic-stress`.
- **Julia style:** match `code/src/SyntheticTables.jl`:
  - a docstring with units on every function;
  - explicit loops;
  - `ArgumentError` for bad input;
  - no closures or `do` blocks in new code.

## Review Focus

1. **Seed-5660 rows counted twice.** The γ and hand-picked rows for seed 5660 sit in both
   `results/gamma_seeds/gamma_seeds_summary.csv` and `results/tier1_summary.csv` or `results/universe_summary.csv`
   (140 shared labels). A cell must have 30 seeds, not 31. Pinned by the duplicate-label tests in Task 1.
2. **The MPC band one day off.** The trigger compares the value τ trading days after a decision with `μ[τ] ± z·σ[τ]`
   (`code/src/Triggers.jl:82`). If the drawn cone is shifted by a day, the band exit can appear inside the band. Pinned
   by the τ-alignment test in Task 2 and by the script's check on the real episode.
3. **Silent font fallback.** `vnslides.sty` falls back to Arial when Helvetica Neue is missing, and the deck still
   compiles. Pinned by the `pdffonts` check in `talk/check_deck.sh` (Task 3).
4. **Slide text that disagrees with the data.** Typed numbers go stale. Pinned by the `talk_numbers` test in Task 1
   and by the macro use in Task 8.
5. **A stylized-fact claim that fails on the data.** "Fat tails, little autocorrelation, volatility clustering" must
   hold for both the real series and the generator. Pinned by the claim checks in `scripts/30_talk_data.jl` (Task 2).
   If a check fails, stop and report it; do not loosen the threshold.

---

## File map

| Path | Role | Task |
|---|---|---|
| `code/src/TalkData.jl` | the data functions behind every talk figure | 1, 2, 5 |
| `code/test/test_talk_data.jl` | their tests | 1, 2, 5 |
| `code/src/Include.jl`, `code/test/runtests.jl` | include, export and test registration | 1 |
| `scripts/30_talk_data.jl` | writes `talk/data/`; checks the slide claims | 1, 2, 5 |
| `talk/informs2026.tex` | the deck: preamble and `\input` of each slide | 3 |
| `talk/vnslides.sty`, `talk/vnflow.sty`, `talk/cornell-seal.pdf` | the adapted style, its palette, the seal | 3 |
| `talk/Makefile`, `talk/check_deck.sh` | build and lint | 3 |
| `talk/slides/*.tex` | one file per slide | 3–8 |
| `talk/figs/*.tex` | TikZ/pgfplots figures | 4–7 |
| `talk/data/*` | generated data, committed | 1, 2, 5 |
| `talk/informs2026.pdf` | the deliverable, committed | 9 |

**Shared dot-table schema (Tasks 1, 4, 5).** Every dot-plot CSV has these columns:
- `row::Int`: 1 is the top row;
- `group_index::Int`: 0 for the first group;
- `label::String`, `group::String`, `allocator::String`;
- `hedge::Bool`;
- `value::Float64`: ending wealth ÷ start, or a percent from 0 to 100.

Each such CSV gets a companion bar CSV with the columns `row`, `group_index`, `label`, `hedge`, `lo` and `hi`.

---

### Task 1: Real-run grid, large-book shares and text numbers

**Files:**
- Modify: `.gitignore`. Add `results/gamma_seeds/runs/`, `results/large_k/runs/` and `results/*/run.log`.
- Commit as they are:
  - `scripts/dev/gamma_seeds.jl`, `scripts/dev/large_k.jl`;
  - `results/gamma_seeds/gamma_seeds_summary.csv`, `results/large_k/large_k_summary.csv`.
- Create: `code/src/TalkData.jl`.
- Modify: `code/src/Include.jl`. Add `include("TalkData.jl")` after `SyntheticTables.jl`, and an `export` line for
  every name below.
- Create: `code/test/test_talk_data.jl`.
- Modify: `code/test/runtests.jl` (append `include("test_talk_data.jl")`).
- Create: `scripts/30_talk_data.jl`.

**Interfaces:**
- Consumes:
  - `picker_key(row)::String` and `TABLE_ALLOCATOR_ORDER` (`["CD","CES","CESw","SG","MinVar","Tangent","LS"]`),
    both in `code/src/SyntheticTables.jl`;
  - `passive_references(md, start, stop; W0)::DataFrame`, with `name` and `terminal` (USD), in `StudyTables.jl`.
- Produces:
  - `TALK_REALRUN_ROWS::Vector{Tuple{String,String,String}}`, one `(picker_key, label, group)` per row in slide
    order:
    - `("gamma", "γ per sector", "33 names · 3 per sector")`
    - `("bandit", "Bandit per sector", "33 names · 3 per sector")`
    - `("random", "Random per sector", "33 names · 3 per sector")`
    - `("global_gamma|K=13", "Global γ", "13 names")`
    - `("corr_gamma|K=13", "Correlation-aware γ", "13 names")`
    - `("liquid_gamma|K=13", "Liquid then γ", "13 names")`
    - `("random_liquid|K=13", "Random, liquid 100", "13 names")`
    - `("random_global|K=13", "Random, whole universe", "13 names")`
    - `("fixed:l8b_13", "Hand-picked 13", "Hand-picked")`
    - `("fixed:l8b_15", "Hand-picked 13 + WMT, PG", "Hand-picked")`
  - `talk_realrun_grid(rows::DataFrame; seeds=5660:5689)::DataFrame`. It returns the dot-table schema plus `picker`
    and `n_seeds`, with `value` = the median of `terminal / W0`, one row per `TALK_REALRUN_ROWS` × hedge
    (false, true) × `TABLE_ALLOCATOR_ORDER`, in that order (140 rows).
    - It uses only `status == "ok"` rows whose seed is in `seeds`.
    - It keeps a duplicated `label` once, but throws `ArgumentError` if the copies' `terminal` values differ by more
      than `1e-9·terminal`.
    - It throws `ArgumentError` if a cell has a seed count other than `length(seeds)`.
  - `talk_row_ranges(dots::DataFrame)::DataFrame`: the bar CSV (`row`, `group_index`, `label`, `hedge`, `lo`, `hi`),
    with the min and max of `value` per row and hedge.
  - `talk_largek_shares(rows::DataFrame; sg_max_K::Int=34)::DataFrame`, with columns `K`, `allocator`, `hedge`,
    `n_global`, `n_random`, `global_median`, `random_median` and `share`.
    - `share` is the fraction of (global, random) seed pairs in which global γ ends strictly higher.
    - It covers screens `global_gamma` and `random_global` only, and drops SG rows with `K > sg_max_K`.
    - It throws `ArgumentError` when a (K, allocator, hedge) has one screen but not the other.
    - Rows are ordered by K, then hedge (false first), then `TABLE_ALLOCATOR_ORDER`.
  - `talk_numbers(grid::DataFrame, n_decisions::AbstractVector{<:Integer}, h2h::DataFrame)::Vector{Pair{String,String}}`,
    where `grid` comes from `talk_realrun_grid` and `h2h` is the `typical_head_to_head.csv` table. It returns macro
    names and their text, in this order:
    - `"SectorLo"`, `"SectorHi"`: min and max of hedge-off γ per sector, 2 decimals;
    - `"BasketLo"`, `"BasketHi"`: min and max of every hedge-off cell, 2 decimals;
    - `"Rebalances"`: `median(n_decisions) − 1`, rounded to an integer;
    - `"TypicalLo"`, `"TypicalHi"`: min and max of `100·share_above_0` over the rows where `comparison` is
      `"Global γ (13) minus Random, whole universe (13)"`, `metric` is `"wealth"` and `hedge` is `"off"`, rounded to
      integers.
  - `write_tex_macros(path::AbstractString, pairs::Vector{Pair{String,String}})::Nothing`. It writes one
    `\newcommand{\<name>}{<text>}` line per pair.

- [ ] **Step 1: Write the failing tests** in `code/test/test_talk_data.jl`, inside `@testset "talk data" begin … end`.
  Build the inputs with explicit loops.

  ```julia
  # a full real-run table: every TALK_REALRUN_ROWS picker × 7 utilities × 2 hedges × 30 seeds,
  # terminal = W0·(1 + 0.001·(seed − 5660)), so every cell's median is 1.0145
  rows = full_rows()                                   # helper defined in the test file
  g = talk_realrun_grid(rows)
  @test nrow(g) == 140
  @test g.label[1] == "γ per sector" && g.hedge[1] == false && g.allocator[1] == "CD"
  @test all(g.n_seeds .== 30) && all(isapprox.(g.value, 1.0145; atol=1e-12))
  @test g.row[1] == 1 && g.group_index[1] == 0 && maximum(g.row) == 10 && maximum(g.group_index) == 2
  # the seed-5660 rows repeated (as in tier1/universe summaries) are counted once
  dup = vcat(rows, rows[rows.seed .== 5660, :])
  @test all(talk_realrun_grid(dup).n_seeds .== 30)
  # a repeated label with a different terminal is an error
  bad = vcat(rows, rows[1:1, :]); bad.terminal[end] *= 1.01
  @test_throws ArgumentError talk_realrun_grid(bad)
  # a missing seed is an error
  @test_throws ArgumentError talk_realrun_grid(rows[2:end, :])
  # bars
  r = talk_row_ranges(g)
  @test nrow(r) == 20 && all(r.lo .<= r.hi)
  # large books: global [1.2, 1.3] vs random [1.1, 1.25, 1.4] → 3 of 6 pairs
  lk = largek_rows()        # K = 34 and 100 × CD and SG, hedge off; each has global [1.2, 1.3], random [1.1, 1.25, 1.4]
  s = talk_largek_shares(lk)
  @test s.share[(s.K .== 34) .& (s.allocator .== "CD")] == [0.5]
  @test !any((s.K .== 100) .& (s.allocator .== "SG")) && any((s.K .== 34) .& (s.allocator .== "SG"))
  @test_throws ArgumentError talk_largek_shares(lk[lk.screen .!= "random_global", :])
  # text numbers
  g2 = copy(g)
  g2.value .= 1.2
  sector_off = (g2.picker .== "gamma") .& (g2.hedge .== false)
  g2.value[sector_off] = [1.086, 1.219, 1.221, 1.234, 1.090, 1.227, 1.110]
  g2.value[(g2.picker .== "random_global|K=13") .& (g2.hedge .== false) .& (g2.allocator .== "LS")] .= 1.031
  g2.value[(g2.picker .== "fixed:l8b_13") .& (g2.hedge .== false) .& (g2.allocator .== "Tangent")] .= 1.697
  h2h = DataFrame(comparison=fill("Global γ (13) minus Random, whole universe (13)", 3), metric="wealth",
      hedge="off", share_above_0=[0.6233, 0.7117, 0.65])
  n = Dict(talk_numbers(g2, [27, 28, 29], h2h))
  @test n["SectorLo"] == "1.09" && n["SectorHi"] == "1.23" && n["BasketLo"] == "1.03" && n["BasketHi"] == "1.70"
  @test n["Rebalances"] == "27" && n["TypicalLo"] == "62" && n["TypicalHi"] == "71"
  ```

- [ ] **Step 2: Run the tests and confirm they fail.**
  Run: `julia --project=code -e 'using Test, Dates, DataFrames, Random, VLINFORMS2026; include("code/test/test_talk_data.jl")'`
  (the test file itself starts with `using DataFrames, Random`)
  Expected: an error that `talk_realrun_grid` is not defined.

- [ ] **Step 3: Implement the five names above in `code/src/TalkData.jl`.**
  - `row` is the index in `TALK_REALRUN_ROWS`. `group_index` counts the distinct groups seen before the row.
  - Keep a picker's display label separate from the synthetic tables' names; Task 5 maps them.

- [ ] **Step 4: Run the tests and confirm they pass.** Same command. Expected: all pass.

- [ ] **Step 5: Write the first part of `scripts/30_talk_data.jl`.**
  - Header and environment: `Pkg.activate(@__DIR__)`, as in `scripts/13_figures.jl`. It writes to
    `talk/data/` (create it).
  - Load the real-run rows from `results/tier1_summary.csv`, `results/universe_summary.csv` and
    `results/gamma_seeds/gamma_seeds_summary.csv`, keeping columns `label`, `status`, `allocator`, `screen`, `seed`,
    `hedge`, `basket_K`, `W0`, `terminal` and `n_decisions`. Then write:
    - `realrun_grid.csv` from `talk_realrun_grid`, and `realrun_bars.csv` from `talk_row_ranges`;
    - `realrun_refs.csv` (`name`, `value`): `passive_references(md, Date(2025,1,2), Date(2026,9,4))` with
      `value = terminal / 100_000`;
    - `largek_shares.csv` from `talk_largek_shares`, using global γ at K = 5–34 from `gamma_seeds_summary.csv`,
      random at K = 5–34 from `universe_summary.csv`, and both at K = 100 and 200 from
      `results/large_k/large_k_summary.csv`;
    - `numbers.tex` from `write_tex_macros(talk_numbers(grid, n_decisions, h2h))`.
      - `n_decisions` comes from the deduplicated real-run rows of the `TALK_REALRUN_ROWS` pickers with seeds
        5660–5689.
      - `h2h` is `results/synthetic/tables/typical_head_to_head.csv`.
  - Print one line per file written.

- [ ] **Step 6: Run the script and check the values.**
  Run: `OMP_NUM_THREADS=1 KMP_GTID_MODE=2 julia scripts/30_talk_data.jl`
  Expected:
  - `realrun_grid.csv` has 140 rows;
  - Global γ CD hedge off is 1.5204;
  - Hand-picked 13 Tangent hedge off is 1.6971;
  - Random, whole universe LS hedge off is 1.0310;
  - Bandit per sector CD hedge on is 1.1248;
  - `realrun_refs.csv` has SPY 1.3174 and equal weight 1.296 (each ±5e-4);
  - `largek_shares.csv` has 94 rows (7 K × 7 utilities × 2 hedges, minus SG at 100 and 200), with K = 100 CD hedge off
    0.7744, K = 200 Tangent hedge on 0.6300 and K = 34 MinVar hedge on 0.2611;
  - `numbers.tex` contains `\newcommand{\SectorLo}{1.09}`, `{\SectorHi}{1.23}`, `{\BasketLo}{1.03}`,
    `{\BasketHi}{1.70}`, `{\Rebalances}{27}`, `{\TypicalLo}{62}` and `{\TypicalHi}{71}`.

- [ ] **Step 7: Commit.**
  ```bash
  git add .gitignore scripts/dev/gamma_seeds.jl scripts/dev/large_k.jl results/gamma_seeds/gamma_seeds_summary.csv \
      results/large_k/large_k_summary.csv code/src/TalkData.jl code/src/Include.jl code/test/test_talk_data.jl \
      code/test/runtests.jl scripts/30_talk_data.jl talk/data
  git commit -m "feat(talk): real-run grid, large-book shares and text numbers for the slides"
  ```
  Check `git status` shows no `runs/` cache files staged.

### Task 2: MPC episode and stylized facts

**Files:**
- Modify: `code/src/TalkData.jl`, `code/test/test_talk_data.jl`, `scripts/30_talk_data.jl`, `code/src/Include.jl`
  (exports).

**Interfaces:**
- Consumes:
  - `MyStudyResult`: `dates`, `equity` (USD), `decisions` (`date`, `reason`), and `projections::Dict{Date,MyProjection}`
    with `μ`, `σ` (USD) for τ = 1..21;
  - `abs_acf(x, lags)` in `SyntheticChecks.jl`;
  - `synthetic_draws(gen, p).g_m`;
  - `growth_window(dates, start, stop)`, where row k is the growth from `dates[k]` to `dates[k+1]`.
- Produces:
  - `talk_mpc_episode(res::MyStudyResult, start::Date, stop::Date; W0::Real=100_000.0)::NamedTuple{(:path, :cones, :events)}`:
    - `path`: `day::Int` (0 at `start`), `date`, and `w = equity / W0`, for `start..stop`;
    - `cones`: `cone::Int` (1 for the cone starting at `start`), `day` (since `start`), and `mean` and `sd` (both ÷ W0).
      - Each cone starts at a decision date in `[start, stop)` with `mean = w` and `sd = 0` at τ = 0.
      - It runs to the next decision date, to τ = 21, or to `stop`, whichever comes first.
    - `events`: `day`, `date`, `reason::String`, and `z = (w − mean)/sd` of the cone active on that day. This covers
      each decision in `(start, stop]`.
    - It throws `ArgumentError` if `start` or `stop` is not in `res.dates`, or `start` is not a decision date with a
      stored projection.
  - `talk_stylized_facts(g_real::AbstractVector{<:Real}, g_syn::AbstractVector{<:AbstractVector{<:Real}}; lags=1:20, edges=-8.0:0.25:8.0)::NamedTuple{(:density, :acf, :abs_acf, :summary)}`.
    Each series is standardized as (g − mean)/sd before the density.
    - `density`: `x` (bin center, SDs), `real`, `syn_p10`, `syn_p50`, `syn_p90` and `normal` (the N(0,1) pdf). Each
      density is count/(n·bin width), and empty bins are `NaN`.
    - `acf`: `lag`, `real`, `syn_p10`, `syn_p50`, `syn_p90`, the autocorrelation of g.
    - `abs_acf`: the same for |g|, via `abs_acf`.
    - `summary`: `series` (`"real"`, `"synthetic median"`), `kurtosis` (m₄/m₂², non-excess), `acf1` and `abs_acf1`.

- [ ] **Step 1: Write the failing tests** (append to the testset).

  ```julia
  # a 12-day result: decisions on day 0 (initial) and day 3 (band_exit); projection from day 0
  # μ = W0·(1 + 0.01τ), σ = 0.01·W0; equity 1.00, 1.005, 1.012, 0.95, … ·W0; second projection from day 3
  res = tiny_result()                                  # helper in the test file, built with build(MyStudyResult, …)
  ep = talk_mpc_episode(res, res.dates[1], res.dates[12])
  @test nrow(ep.path) == 12 && ep.path.w[4] ≈ 0.95
  c1 = ep.cones[ep.cones.cone .== 1, :]
  @test c1.day == [0, 1, 2, 3] && c1.sd[1] == 0.0 && c1.mean[4] ≈ 1.03     # τ aligned: day 3 uses μ[3]
  @test ep.events.reason == ["band_exit"] && ep.events.day == [3] && ep.events.z[1] ≈ (0.95 - 1.03) / 0.01
  @test ep.cones[ep.cones.cone .== 2, :].mean[1] ≈ 0.95
  @test_throws ArgumentError talk_mpc_episode(res, res.dates[2], res.dates[12])
  # stylized facts on iid normal draws and on a clustered series
  x = randn(MersenneTwister(1), 20_000)
  sf = talk_stylized_facts(x, [x, x])
  @test abs(sf.acf.real[1]) < 0.03 && abs(sf.summary.kurtosis[1] - 3) < 0.15 && abs(sf.abs_acf.real[1]) < 0.03
  w = diff(collect(sf.density.x))[1]
  @test isapprox(sum(filter(!isnan, sf.density.real)) * w, 1.0; atol=0.01)
  σt = [isodd(div(t - 1, 50)) ? 3.0 : 1.0 for t in 1:20_000]
  y = σt .* randn(MersenneTwister(2), 20_000)
  sy = talk_stylized_facts(y, [y])
  @test sy.abs_acf.real[1] > 0.15 && sy.summary.kurtosis[1] > 3.5
  ```

- [ ] **Step 2: Run the tests and confirm the new ones fail.** Same command as Task 1. Expected: an error that
  `talk_mpc_episode` is not defined.

- [ ] **Step 3: Implement both functions** in `code/src/TalkData.jl`.
  - Cone τ indexing must match `check_trigger`: the value τ trading days after the decision is compared with
    `μ[τ]`.
  - Percentiles across synthetic series are taken bin by bin and lag by lag, ignoring `NaN`.

- [ ] **Step 4: Run the tests and confirm they pass.**

- [ ] **Step 5: Extend `scripts/30_talk_data.jl`.**
  - **MPC episode.**
    - Find the `hash` of label `CD|fixed:l8b_13|hedge=false|seed=5660` in `gamma_seeds_summary.csv`, load
      `results/gamma_seeds/runs/<hash>.jld2` key `"result"`, and call
      `talk_mpc_episode(res, Date(2025,10,10), Date(2025,12,22))`.
    - Check with `error` if any fails:
      - the event reasons are `["horizon_elapsed", "band_exit", "horizon_elapsed"]`;
      - the band-exit `|z| > 1.96`;
      - on every cone day strictly between a cone's start and its last day, `|z| ≤ 1.96`.
    - Write `mpc_path.csv`, `mpc_events.csv` and one `mpc_cone_<k>.csv` per cone (`day`, `mean`, `sd`).
  - **Stylized facts.**
    - Real: SPY growth `log(S[k+1]/S[k])·252` for k in `growth_window(md.dates, Date(2014,1,3), Date(2024,12,31))`.
    - Synthetic: load `results/synthetic/generator.jld2` key `"generator"`. Take the 100 lowest `path_id` with
      `typical == true` in `results/synthetic/paths.csv`, and for each,
      `synthetic_draws(gen, p).g_m[growth_window(gen.dates, Date(2014,1,3), Date(2024,12,31))]`.
    - Write `sf_density.csv`, `sf_acf.csv`, `sf_abs_acf.csv` and `sf_summary.csv`.
    - **Claim checks.** If any fails, `error` with the values, then stop and report; do not change a threshold. For
      both `"real"` and `"synthetic median"`:
      - kurtosis > 4;
      - |acf at lag 1| < 0.15;
      - the abs ACF at lags 1–5 all > 0.10.

- [ ] **Step 6: Run the script.** Same command as Task 1.
  Expected:
  - three `mpc_cone_*.csv` files: cone 1 has days 0–21, cone 2 days 21–29, cone 3 days 29–50;
  - the band-exit z is about −2.10;
  - the claim checks pass and `sf_summary.csv` is printed.

- [ ] **Step 7: Commit.**
  `git commit -m "feat(talk): MPC episode and stylized-fact data for the slides"`, with the changed files and
  `talk/data`.

### Task 3: Deck skeleton, check script and acknowledgements

**Files:**
- Create:
  - `talk/vnslides.sty`: a copy of
    `~/Desktop/julia_work/CHEME-5660-CourseRepository-Fall-2026/lectures/templates/vnslides.sty`, with the edits
    below;
  - `talk/vnflow.sty`: a copy of the template's `vnflow.sty`;
  - `talk/cornell-seal.pdf`: a copy of `templates/assets/cornell-seal.pdf`.
- Create: `talk/informs2026.tex`, `talk/Makefile` and `talk/check_deck.sh`.
- Create placeholder slides: `talk/slides/02-utility.tex`, `03-mpc.tex`, `04-jumphmm.tex`, `05-realrun.tex`,
  `06-synthetic.tex`, `07-conclusions.tex`, `b1-percentiles.tex`, `b2-large-books.tex`, `b3-tables-off.tex` and
  `b3-tables-on.tex`. Each is one frame with its placeholder title and no other content.
- Create the full acknowledgements slide: `talk/slides/08-acknowledgements.tex`.
- Modify: `.gitignore` for the build products (`talk/*.aux`, `*.log`, `*.nav`, `*.out`, `*.snm`, `*.toc`,
  `*.fdb_latexmk`, `*.fls`, `*.xdv`, `talk/build/`).

**Interfaces:**
- Produces:
  - `make -C talk`, which builds `talk/informs2026.pdf` with `latexmk -xelatex`.
  - `talk/check_deck.sh <expected_pages>`. It exits non-zero on any of:
    - a build error;
    - `Overfull` in the log;
    - an undefined reference;
    - `pdffonts` not listing `HelveticaNeue-Light`;
    - a page count other than the argument;
    - any of these strings in `talk/slides/*.tex` or `talk/figs/*.tex`: `---`, `global_gamma`, `l8b`, `fixed:`,
      `SCS`, `TODO`, `TBD`, or (case-insensitive) `returns` and `hindsight`.

    It then writes page PNGs with `pdftoppm -r 60 -png` to `talk/build/png/`.
  - `\input{data/numbers.tex}` in the preamble.

- [ ] **Step 1: Write `talk/check_deck.sh`** (bash, `set -euo pipefail`), with the behavior above.

- [ ] **Step 2: Run it before the deck exists and confirm it fails.** Run `bash talk/check_deck.sh 12`. Expected: a
  non-zero exit because the build fails.

- [ ] **Step 3: Adapt `talk/vnslides.sty`** with these edits only.
  - Header comment: say it is adapted from the 5660 template for the INFORMS talk.
  - `\input{vnflow.sty}` (local) and `\vnseal{cornell-seal.pdf}`.
  - The frame title is followed by a coral rule, `\textcolor{vnaccent}{\rule{0.072\paperwidth}{1.4pt}}`, after
    0.35em.
  - The footline is the frame number only, bottom right, at 7pt in `vnslidebody` grey. It replaces the Cornell
    lockup.
  - `\vntitlepage` uses `\begin{frame}[plain]`.
  - Add `\RequirePackage{pgfplots}`, `\pgfplotsset{compat=1.18}`, `\usepgfplotslibrary{fillbetween}`,
    `\RequirePackage{pgfplotstable}` and `\RequirePackage{qrcode}`.

- [ ] **Step 4: Write `talk/informs2026.tex`.**
  - Preamble: `\documentclass[aspectratio=169,11pt]{beamer}`, `\usepackage{vnslides}`,
    `\input{data/numbers.tex}`.
  - `\title{Coupling Interpretable Allocation, Model-Predictive Control, and Tail-Aware Forecasting for Live Portfolio Management}`
  - `\subtitle{INFORMS Annual Meeting 2026 · San Francisco · Session TD47 · November 3, 2026}`
  - `\author{Jeffrey D. Varner · Cornell University}`
  - Body: `\vntitlepage`, then an `\input` of each slide file in order 02–08, then `\appendix` and the four backup files.

- [ ] **Step 5: Write `talk/slides/08-acknowledgements.tex`.**
  - Title: "Acknowledgements".
  - Left two-thirds, the people:
    - **Abdulrahman Alswaidan**, MS 2026: co-author on the JumpHMM paper.
    - **Aaron Wheeler**, PhD 2024: helpful discussions.
    - **Evan Vu**, undergraduate, class of 2026, now at Bridgewater: an early version of the Cobb-Douglas allocation
      and the bandit picker.
  - Then a line "Alpaca Markets", in plain text.
  - Right third: two QR codes stacked or side by side, each about 2.4cm, via `\qrcode[hyperlink,height=2.4cm]{…}`:
    - "Paper": `https://arxiv.org/abs/2603.10202`, captioned "JumpHMM paper · arXiv:2603.10202";
    - "Code": `https://github.com/varnerlab/INFORMS-Annual-Meeting-2026-Varnerlab`, captioned
      "github.com/varnerlab/INFORMS-Annual-Meeting-2026-Varnerlab" in footnote size.

- [ ] **Step 6: Build and check.** Run `bash talk/check_deck.sh 12`. Expected: exit 0, 12 pages.
  - Open `talk/build/png/` pages 1 and 8 and confirm that:
    - the title page shows the seal, title, subtitle and author, with no slide number;
    - page 8 shows a coral rule under the title, the three people, Alpaca, two QR codes, and "8" bottom right.

- [ ] **Step 7: Commit.** `git commit -m "feat(talk): deck skeleton, check script and acknowledgements"`, with `talk/`
  (not the build products) and `.gitignore`.

### Task 4: Dot-plot figure, real-run slide and number tables

**Files:**
- Create: `talk/figs/dotplot.tex`.
- Modify: `talk/slides/05-realrun.tex`, `talk/slides/b3-tables-off.tex` and `talk/slides/b3-tables-on.tex`.
- Modify: `scripts/30_talk_data.jl`, to write `b3_rows_hedge_off.tex` and `b3_rows_hedge_on.tex`.

**Interfaces:**
- Consumes: `talk/data/realrun_grid.csv`, `realrun_bars.csv` and `realrun_refs.csv` (Task 1).
- Produces: a macro `\vndotplot[<keys>]{<dots.csv>}{<bars.csv>}` that draws one dot-plot panel in the grammar;
  Task 5 reuses it. The options are pgfkeys, because a TeX macro takes at most 9 arguments:
  - `xmin`, `xmax`, `xstep`;
  - `spy` (coral dashed line), `ew` (grey dotted line) and `midline` (grey dashed line, e.g. 50), each empty by
    default, which means no line;
  - `width`, `height`;
  - `rowlabels` (`true` by default).

- [ ] **Step 1: Implement `\vndotplot` in `talk/figs/dotplot.tex`.** Use pgfplots plus `pgfplotstable`, applying
  every dot-plot rule in Global Constraints, with this geometry:
  - row y = −(row + 0.7·group_index);
  - hedge off at y + 0.13 (filled #579ACA, radius 2.2pt);
  - hedge on at y − 0.13 (white fill, #2F6FAE 0.8pt outline, radius 2pt);
  - bars at the same offsets, #DBE7F3, line width 2.6pt, round caps;
  - group labels above each group's first row: uppercase, 6pt, link blue, letter-spaced;
  - row labels at 9pt in ink;
  - SPY: coral, 1.1pt, dash pattern on 4pt off 2.5pt; equal weight: #A8A8A8, 1pt, dotted.

  The legend is a separate macro `\vndotlegend`, one line under the plot.

- [ ] **Step 2: Write the real-run slide `05-realrun.tex`.**
  - Title: "What you hold beats how you weight it".
  - The line under the title: "Real-data backtest, Jan 2025 – Sep 2026".
  - `\vndotplot` with x from 0.95 to 1.75, step 0.1, and SPY and equal weight taken from `realrun_refs.csv`. Read
    them with `pgfplotstable` rather than typing them.
  - X axis label: "ending wealth ÷ start, median of 30 runs".
  - The legend.
  - The hedge gloss in footnote size: "hedge on: up to 5 names with negative β join the basket and the book's β is
    capped".
  - The footnote: "A paper-trading deployment of this system is running; it is not part of this talk."

- [ ] **Step 3: Write the number tables.** In `scripts/30_talk_data.jl`, add `b3_rows_hedge_{off,on}.tex`: one
  `label & v_CD & … & v_LS \\` line per `TALK_REALRUN_ROWS` row, then SPY and equal weight (the same value in every
  column), at 3 decimals. Rerun the script.
  - `b3-tables-off.tex` title: "Show me the numbers: hedge off". It holds a `tabular` with the header
    `Picker & CD & CES & CESw & SG & MinVar & Tangent & LS`, `\input{data/b3_rows_hedge_off.tex}`, and the note
    "Ending wealth ÷ start, median of 30 runs. Real-data backtest, Jan 2025 – Sep 2026."
  - `b3-tables-on.tex` is the same for hedge on.

- [ ] **Step 4: Build, check and look.**
  - Run `bash talk/check_deck.sh 12`. Expected: exit 0.
  - View the page PNGs of the real-run slide (page 5) and both table slides (pages 11 and 12) at full size. Confirm:
    - 10 rows in 3 groups, 14 dots per row, with filled dots above and hollow below;
    - the coral dashed SPY line at 1.317 and the grey dotted equal-weight line at 1.296;
    - no text collides;
    - the table rows match `realrun_grid.csv`, for example Global γ CD hedge off reads 1.520.

- [ ] **Step 5: Commit.** `git commit -m "feat(talk): real-run dot plot and number tables"`.

### Task 5: Synthetic slide, percentile and large-book backups

**Files:**
- Modify: `code/src/TalkData.jl`, `code/test/test_talk_data.jl`, `scripts/30_talk_data.jl`, `code/src/Include.jl`.
- Modify: `talk/slides/06-synthetic.tex`, `talk/slides/b1-percentiles.tex` and `talk/slides/b2-large-books.tex`.

**Interfaces:**
- Consumes: `PICKER_ROWS` (synthetic table names by picker key), `TALK_REALRUN_ROWS`, `talk_row_ranges` and
  `\vndotplot`.
- Produces:
  - `talk_from_wide(t::DataFrame; hedge::Bool, scale::Real=1.0)::NamedTuple{(:dots, :refs)}`, converting a wide
    synthetic table (column `Ticker picker` plus the seven utility columns):
    - `dots`: the dot-table schema for the `TALK_REALRUN_ROWS` pickers, with names matched through `PICKER_ROWS`
      and `value = scale·cell`;
    - `refs`: `name`, `value` for `"SPY buy-and-hold"` and `"equal-weight universe"`.
    - It throws `ArgumentError` if a picker's row is missing.
  - `talk_largek_dots(shares::DataFrame)::DataFrame`: the dot-table schema with one row per K in increasing order,
    label `"<K> names"`, group `"Global γ against random picks"`, and `value = 100·share`.

- [ ] **Step 1: Write the failing tests.**
  - `talk_from_wide` on a hand-built wide table:
    - the 14 `PICKER_ROWS` names plus the two reference rows, 7 utility columns, every cell 1.1;
    - expect 70 dot rows for one hedge setting, with labels in slide order and the "Global γ (5)" row dropped;
    - `refs.value == [1.1, 1.1]`;
    - with `scale = 100`, values of 110;
    - with "Global γ (13)" removed, an `ArgumentError`.
  - `talk_largek_dots` on `talk_largek_shares` output with K = 34 and 100: rows 1 and 2, labels `"34 names"` and
    `"100 names"`, and value `100·share`.

- [ ] **Step 2: Run the tests and confirm they fail; implement; run them and confirm they pass.** Same command as
  Task 1.

- [ ] **Step 3: Extend the script and rerun it.**
  - For each set in `typical`, `burst_stress` and `bear_stress`: read `<set>_hedge_{off,on}_wealth_median.csv`, then
    write `syn_<set>_dots.csv` (both hedge settings stacked), `syn_<set>_bars.csv` and `syn_<set>_refs.csv` (hedge
    off). Also write `syn_axes.csv` (`set`, `lo`, `hi`), where over the set's dots and references
    lo = ⌊(min − 0.005)·20⌋/20 and hi = ⌈(max + 0.005)·20⌉/20.
  - B1: from `typical_real_percentile_hedge_{off,on}.csv` with `scale = 100`, write `b1_dots.csv` and
    `b1_bars.csv`. Write `b1_refs.csv` with SPY's percentile, which should be 70.
  - B2: write `b2_dots.csv` and `b2_bars.csv` from `talk_largek_dots(largek_shares)`.

- [ ] **Step 4: Write the synthetic slide `06-synthetic.tex`.**
  - Title: "978 synthetic futures in three regimes".
  - Three `\vndotplot` panels:
    - row labels on the first panel only;
    - per-panel x ranges from `syn_axes.csv`, ticks every 0.1;
    - panel headers "Typical" / "600 futures", "Burst stress" / "200 futures", "Bear stress" / "200 futures";
    - each panel's own SPY and equal-weight lines.
  - Shared axis title: "ending wealth ÷ start, median across futures". The legend sits once, under the panels.

- [ ] **Step 5: Write `b1-percentiles.tex`.**
  - Title: "Was 2025 a lucky year?".
  - One `\vndotplot` with x from 0 to 100, step 25, `midline=50`, no equal-weight line, and `spy` set to SPY's
    percentile from `b1_refs.csv`. That is the usual coral dashed SPY line, at 70.
  - X label: "percentile of the real result among 600 typical futures".
  - Footnote: "Real results use the same seeds as the simulated runs: 5660–5664 for random and bandit pickers, 5660 for
    the others."

- [ ] **Step 6: Write `b2-large-books.tex`.**
  - Title: "Does this work for a real-sized book?".
  - One `\vndotplot`: rows for 5, 8, 13, 21, 34, 100 and 200 names; x from 0 to 100, step 25; a 50% line; no SPY or
    equal-weight line.
  - X label: "share of seed pairs in which global γ ends higher than random picks (%)".
  - Footnote: "Real-data backtest, 30 runs each. Stone–Geary left out at 100 and 200 names: its 1%-per-name floor uses
    the whole budget at 100 names and twice the budget at 200."

- [ ] **Step 7: Build, check and look.**
  - Run `bash talk/check_deck.sh 12`. Expected: exit 0.
  - View pages 6, 9 and 10. Confirm:
    - all three panels have 10 rows and their own SPY lines (typical SPY about 1.185, burst about 0.865, bear about
      0.718);
    - B1's SPY row is at 70;
    - B2 shows 6 dots per hedge at 100 and 200 names and 7 elsewhere.

- [ ] **Step 8: Commit.** `git commit -m "feat(talk): synthetic three-panel slide, percentile and large-book backups"`.

### Task 6: MPC slide

**Files:**
- Create: `talk/figs/mpc.tex`.
- Modify: `talk/slides/03-mpc.tex`.

**Interfaces:**
- Consumes: `talk/data/mpc_path.csv`, `mpc_events.csv` and `mpc_cone_{1,2,3}.csv` (Task 2), and `\Rebalances`.

- [ ] **Step 1: Write `talk/figs/mpc.tex`**, one pgfplots axis.
  - Plot area: `gray95` background, white grid lines.
  - x from 0 to 50, with ticks at the event days and 0, labeled "Oct 10", "Nov 10", "Nov 20" and "Dec 22". X label
    "2025, trading days".
  - y label `$W/W_0$`, ticks every 0.05.
  - For each cone, `fillbetween` bands of mean ± z·sd for z = 2.576, 1.96 and 1.0, in `deepskyblue1` at fill opacity
    0.13, 0.17 and 0.24, layered.
  - Labels "99%", "95%" and "68%" just right of the last cone's end.
  - The realized path from `mpc_path.csv`: red #FF0000, 1.8pt, round joins.
  - At each event, a coral ring marker (white fill, #FF644E 1.2pt) with labels, kept clear of the path and bands:
    - "Nov 10: 21 days up → re-plan" above;
    - "Nov 20: below the 95% band → re-plan" below;
    - "Dec 22: 21 days up → re-plan" above and left.

- [ ] **Step 2: Write `03-mpc.tex`.**
  - Title: "When to rebalance: re-plan when the market leaves the forecast".
  - Line 1 under the title: "Controller's model: multi-asset GBM, 1,000 paths, 21 trading days. Plant: real prices."
  - Line 2: "Re-plan when the value leaves the 95% band, falls 8% below its peak, or 21 trading days pass."
  - The figure.
  - Footnote: "Real-data backtest: hand-picked 13 names, Cobb-Douglas, hedge off. Bands are the forecasts stored at
    each rebalance. Over the full backtest: about \Rebalances{} rebalances after the first allocation in 419 trading
    days."

- [ ] **Step 3: Build, check and look.**
  - Run `bash talk/check_deck.sh 12`. Expected: exit 0.
  - View page 3 and compare it with the spec's MPC section. Confirm:
    - three cones;
    - the second ends at Nov 20 where the red line is below the outer edge of the middle band;
    - no label overlaps the red line.

- [ ] **Step 4: Commit.** `git commit -m "feat(talk): MPC trigger slide from a real episode"`.

### Task 7: JumpHMM slide

**Files:**
- Create: `talk/figs/jumphmm_architecture.tex` and `talk/figs/stylized_facts.tex`.
- Modify: `talk/slides/04-jumphmm.tex`.

**Interfaces:**
- Consumes: `talk/data/sf_density.csv`, `sf_acf.csv` and `sf_abs_acf.csv` (Task 2).
- Source: the `tikzpicture` of `fig:model_architecture` in
  `~/Desktop/julia_work/HMM-w-jumps-paper/arxiv-paper/sections/main_figures.tex` (around lines 21–109).

- [ ] **Step 1: Restyle the architecture** into `talk/figs/jumphmm_architecture.tex`. Keep the structure and the math
  labels, and change only the colors and fonts:
  - bottom tail: red → coral #FF644E (fill at 25%);
  - top tail: blue → link blue #579ACA (fill at 25%);
  - Poisson clock: yellow fill → white with ink border;
  - grey arrows and notes → body grey;
  - fonts inherit the slide font.

  Wrap it in `\resizebox{\linewidth}{!}{…}`.

- [ ] **Step 2: Write `talk/figs/stylized_facts.tex`**, three small stacked pgfplots axes of equal width with
  `gray95` backgrounds.
  - (a) log10 density over x from −8 to 8 SDs: the synthetic p10–p90 band in `deepskyblue1` at opacity 0.25, the
    synthetic median in #579ACA at 0.8pt, the real line in red #FF0000 at 1.2pt, and N(0,1) dashed in grey. Skip
    `NaN` bins.
  - (b) ACF of g, lags 1–20: the synthetic band and the real line.
  - (c) ACF of |g|: the same.

  Panel titles: "fat tails", "little autocorrelation", "volatility clustering".

- [ ] **Step 3: Write `04-jumphmm.tex`.**
  - Title: "JumpHMM: a market generator with fat tails and volatility clustering".
  - Left half: the architecture, with "Alswaidan and Varner, arXiv:2603.10202" under it in footnote size.
  - Right half: the stylized facts.
  - Caption line: "Real SPY daily growth, 2014–2024 (red), against 100 paths of this year's generator (blue, 10th to
    90th percentile)."

- [ ] **Step 4: Build, check and look.**
  - Run `bash talk/check_deck.sh 12`. Expected: exit 0.
  - View page 4. Confirm:
    - the schematic reads at slide size;
    - the real density sits above the normal curve in both tails;
    - the ACF of |g| is clearly above zero at the first lags.

- [ ] **Step 5: Commit.** `git commit -m "feat(talk): JumpHMM slide with architecture and stylized facts"`.

### Task 8: Utility slide and conclusions

**Files:**
- Modify: `talk/slides/02-utility.tex` and `talk/slides/07-conclusions.tex`.

**Interfaces:**
- Consumes: the macros in `talk/data/numbers.tex`: `\SectorLo`, `\SectorHi`, `\BasketLo`, `\BasketHi`, `\Rebalances`,
  `\TypicalLo` and `\TypicalHi`.

- [ ] **Step 1: Write `02-utility.tex`**, one frame with two builds.
  - Title: "Utility functions turn a preference score into weights".
  - The 5660 house rule applies: define every symbol in the prose before its display.
  - Build 1 (`\only<1>`):
    - prose defining shares nᵢ, prices Sᵢ, wealth W and weights wᵢ = nᵢSᵢ/W;
    - the preference weight $\gamma_i = \tanh\big((\alpha_i + \beta_i\tilde g_M)/\max(|\beta_i|,\beta_{\text{floor}})^{\xi}\big)$,
      with αᵢ and βᵢ from a single-index model of daily growth rates and g̃_M the market growth signal;
    - "choose n to maximize U(n; γ) subject to:" followed by one list: the budget; at most 10% of wealth per name; a
      cap on the portfolio's growth-rate volatility, set to SPY's 2014–2024 level; a spread-cost turnover budget per
      rebalance; with the hedge on, a cap on the portfolio β;
    - the four utilities:
      - Cobb-Douglas $\sum \gamma_i \ln n_i$;
      - CES $(\sum \gamma_i n_i^{\rho})^{1/\rho}$ with $\rho = 1 - 1/\eta$;
      - CES on weights, the same with $w_i$;
      - Stone–Geary $\sum \gamma_i \ln(n_i - b_i)$;

      each sum over $\gamma_i > 0$;
    - "Each solve is a convex cone program: exponential cones for the log utilities, power cones for CES."
  - Build 2 (`\only<2>`): keep the γ line. Then "Under the budget alone, the closed forms:":
    - Cobb-Douglas $w_i \propto \gamma_i$;
    - CES on weights $w_i \propto \gamma_i^{\eta}$;
    - CES on shares $n_i \propto (\gamma_i/S_i)^{\eta}$.

    Then "No solver needed, so allocation is instant." Then a bottom line in both builds: "Also compared: minimum
    variance, the tangency portfolio, and 130/30 long-short Cobb-Douglas."

- [ ] **Step 2: Write `07-conclusions.tex`.**
  - Title: "Key takeaways".
  - Exactly three bullets, the approved text from the spec's slide 7, verbatim except that the numbers are macros:
    - "between \SectorLo{} and \SectorHi{} times the starting wealth, while different baskets ranged from \BasketLo{}
      to \BasketHi{}";
    - "about \Rebalances{} rebalances in 419 trading days";
    - "in \TypicalLo–\TypicalHi\% of typical markets".
  - Use `\small` if needed to fit. Do not add the large-book sentence.

- [ ] **Step 3: Build, check and look.**
  - Run `bash talk/check_deck.sh 13`. Expected: exit 0, 13 pages; the utility frame is pages 2–3.
  - View pages 2, 3 and 8. Confirm:
    - build 2 replaces the problem with the closed forms;
    - the takeaways read 1.09, 1.23, 1.03, 1.70, 27 and 62–71%.

- [ ] **Step 4: Commit.** `git commit -m "feat(talk): utility slide and conclusions"`.

### Task 9: Whole-deck check and the PDF

**Files:**
- Commit: `talk/informs2026.pdf`.

- [ ] **Step 1: Run the full test suite.** Run `julia --project=code -e 'using Pkg; Pkg.test()'`. Expected: all pass.

- [ ] **Step 2: Rerun the data script, then check the deck.**
  - Run `OMP_NUM_THREADS=1 KMP_GTID_MODE=2 julia scripts/30_talk_data.jl` and then `bash talk/check_deck.sh 13`.
  - Expected: `git status --short talk/data` is empty, because the data is reproducible, and the check exits 0.

- [ ] **Step 3: Review every file under `talk/`** (`git ls-files talk`). List them in the report. For each slide file,
  check it against its spec section and the Global Constraints. View all 13 page PNGs at full size.

- [ ] **Step 4: Commit the PDF.** `git add -f talk/informs2026.pdf` and
  `git commit -m "build(talk): INFORMS 2026 deck PDF"`. Do not push.
