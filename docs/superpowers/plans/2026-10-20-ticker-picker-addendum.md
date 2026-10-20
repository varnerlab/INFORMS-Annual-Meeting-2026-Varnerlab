# Ticker-Picker Addendum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add universe-wide ticker screens (global γ, liquid-then-γ, correlation-aware γ), their random controls, and the user's fixed lists. Run them at Fibonacci basket sizes K ∈ {5, 8, 13, 21, 34}, with a per-stock cap that scales for screen picks only. Then rerun the base study as a regression check and report the size sweep.

**Architecture:**
- The work extends the built `VLINFORMS2026` package on branch `feat/screened-panel`.
- **Allocators:** they gain a per-name per-stock cap vector, which defaults to the scalar cap, so every existing problem stays bit-identical.
- **Screens:** the new screens follow the existing `AbstractScreen` / `pick` pattern and carry their own K.
- **Backtest:** `run_study` gives screen picks the scaled cap.
- **Runner and tables:** they gain the new run lists and a generalized matched-control percentile.
- **Scripts:** they add a `--universe` sweep, a regression check against the committed base summaries, and the size-sweep tables and figures.

**Tech Stack:** Julia 1.12; JuMP + Clarabel (SCS fallback); DataFrames, CSV, JLD2, TOML, Statistics, Random, Dates; Plots.jl (scripts environment); Test.

**Spec:**
- Addendum (binding for this plan): `docs/superpowers/specs/2026-10-20-ticker-picker-addendum-design.md`.
- Base spec: `docs/superpowers/specs/2026-10-20-screened-utility-panel-study-design.md`.
- § numbers below refer to the addendum unless marked "base".

## Global Constraints

- **Growth rates only (user):** g = ln(S_end/S_start)/(k·Δt) in 1/yr, Δt = 1/252. Nothing is computed, named or labeled as a "return".
- **No look-ahead:** a pick on day t uses data dated ≤ t. Every new screen that reads prices has a `perturb_after` canary test.
- **No tuning on the run window (addendum §2):** the thresholds are fixed before any run. They are `liquidity_shortlist = 100`, `liquidity_window = 63`, `corr_max = 0.7`, `corr_window = 252`, `w_max_scale = 1.5`, and K ∈ {5, 8, 13, 21, 34}.
- **The user's lists are not hindsight-biased:**
  - Main list: AAPL, MSFT, INTC, MU, AMD, NVDA, GS, BAC, WFC, C, F, GM, JNJ.
  - Sensitivity list: the same 13 plus WMT and PG.
  - The user chose both in advance without 2025 data. Never describe them otherwise in code, docs or captions.
  - The new screens are exploratory, because they were designed after Tier 1. Captions and the README say so.
- **Per-stock cap (§5):**
  - Screen picks get `max(w_max, w_max_scale/K)`, where K is the pick count before the hedge pool.
  - Hedge-pool-only names and 130/30 short names get `w_max`.
  - A pick that is also in the pool gets the pick's cap.
- **Existing behavior is unchanged:** when every cap equals the scalar, every allocator result is bit-identical to before. The rerun base summaries must match the committed `results/tier1_summary.csv` and `results/fixed_summary.csv` metric by metric.
- **Conventions:** follow the plan-1 conventions:
  - `My*` types with `build`;
  - docstrings with units;
  - seeded RNGs;
  - ASCII settings fields equal to the TOML keys;
  - new test files added to `code/test/runtests.jl`;
  - single test files run with `cd code/test && julia --project=.. test_<name>.jl`.
- **Commit trailer:** every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **The correlation pass accepts fewer than K names.** Expected: fill the remainder from the highest-γ names not yet accepted, and always return exactly min(K, universe) names. Test in Task 2.
2. **A fixed-list ticker is missing from the universe.** Expected: `ArgumentError` naming the ticker, not a silent drop. Test in Task 2.
3. **A name's correlation is NaN** (zero-variance growth over the window). Expected: treat it as not exceeding `corr_max`, so the name is accepted, and no error. Test in Task 2.
4. **A pool name that is also a screen pick, and a pool-only name.** Expected: the pick's scaled cap for the first, and `w_max` for the second. Test in Task 3.
5. **Rerunning the base grid after the changes.** Expected: summaries identical to the committed ones. Task 1's bit-identical test covers the allocator side; Task 5 checks it end to end.

---

### Task 1: Per-name per-stock caps

**Files:**
- Modify:
  - `code/src/Allocators.jl` (`MyConstraintSet`, its `build`, `_check_problem`, `add_shared_constraints!`)
  - `code/src/AllocatorsMeanVariance.jl` (`_without_vol_cap`)
  - `code/test/helpers.jl` (`P`: add the `w_cap=nothing` keyword)
- Test: `code/test/test_allocators_caps.jl`

**Interfaces:**
- Produces:
  - `MyConstraintSet` gains the field `w_cap::Vector{Float64}`. It is a per-name cap on |wᵢ| (dimensionless) and is optional in `build`.
    - Default: `fill(w_max, K)`, with K = `length(sector)`.
    - Validation: length must be K (`DimensionMismatch`); every entry finite and > 0 (`ArgumentError`).
  - `add_shared_constraints!` uses `c.w_cap[i]` wherever it used `c.w_max`:
    - the per-name rows, long-only and signed;
    - the signed turnover guard `max_l1`.
  - `_without_vol_cap` copies `w_cap`.
  - The `w_max` field stays, as the scalar default, and keeps its docstring meaning.

- [ ] **Step 1: Write the failing tests** (`test_allocators_caps.jl`)

```julia
using Test, LinearAlgebra, VLINFORMS2026
include("helpers.jl")
const RULES = (MyCobbDouglasRule(), build(MyCESRule, (η=0.5,)), build(MyCESRule, (η=2.0,)),
    build(MyStoneGearyRule, (baseline_frac=0.01,)), MyMinVarRule(), MyTangentRule(), MyLongShortCDRule())
@testset "default per-name caps are bit-identical" begin
    for rule in RULES
        a = allocate(rule, P(w_max=0.4, μ_g=[0.3, 0.2, 0.1]))
        b = allocate(rule, P(w_max=0.4, μ_g=[0.3, 0.2, 0.1], w_cap=[0.4, 0.4, 0.4]))
        @test a.n == b.n
        @test a.status == b.status
    end
end
@testset "a per-name cap binds on its own name" begin
    r = allocate(MyCobbDouglasRule(), P(w_max=1.0, w_cap=[1.0, 0.3, 1.0]))
    @test r.w[2] ≈ 0.3 rtol=1e-4
    @test r.w[[1, 3]] ≈ [0.28, 0.42] rtol=1e-4                   # the rest split 0.7 by γ (0.2 : 0.3)
    m = allocate(MyMinVarRule(), P(w_max=1.0, w_cap=[0.2, 1.0, 1.0]))
    @test m.w[1] <= 0.2 + 1e-6
    @test m.w[2:3] ≈ [0.4, 0.4] rtol=1e-4                       # Σ = 4I: equal weights where uncapped
    l = allocate(MyLongShortCDRule(), P5(γ=[0.5, 0.3, 0.2, -0.4, -0.1], w_max=0.6, w_cap=[0.6, 0.6, 0.6, 0.05, 0.6]))
    @test abs(l.w[4]) <= 0.05 + 1e-6
end
@testset "w_cap validation" begin
    base = (w_max=0.1, s_max=1.0, σ_g_max=1.0, K_turnover=1.0, c̄=0.0, sector=["A", "B"], n_prev=[0.0, 0.0])
    @test_throws DimensionMismatch build(MyConstraintSet, merge(base, (w_cap=[0.1],)))
    @test_throws ArgumentError build(MyConstraintSet, merge(base, (w_cap=[0.1, 0.0],)))
    @test build(MyConstraintSet, base).w_cap == [0.1, 0.1]
end
```

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_allocators_caps.jl`. Expected: FAIL (`w_cap` keyword unknown to `P`, or the field is missing).
- [ ] **Step 3: Implement** the Interfaces above. Add `test_allocators_caps.jl` to `runtests.jl`.
- [ ] **Step 4: Run** the new file, then the full suite: `julia --project=code code/test/runtests.jl`. Expected: all PASS, with 1023+ old assertions unchanged.
- [ ] **Step 5: Commit** `git commit -am "feat(allocators): per-name per-stock caps (default = scalar, bit-identical)"` (after `git add` of the new test file).

---

### Task 2: Universe-wide screens and the fixed-list screen

**Files:**
- Modify: `code/src/Screens.jl`
- Test: `code/test/test_screens_universe.jl`

**Interfaces:**
- Consumes: `_stream_seed`, `_sample_without_replacement` (Bandit.jl); `pick`'s existing signature.
- Produces:
  - `liquid_shortlist(md::MyMarketData, t::Int; size::Int=100, window::Int=63)::Vector{Int}`.
    - The `size` columns with the largest median dollar volume, the median of `close[k, j]·volume[k, j]` (USD/day) over rows k ∈ max(1, t−window+1):t.
    - Ranked by dollar volume descending, ties by column index.
  - Screen types, all `<: AbstractScreen`. Each has positional outer constructors in the field order listed, plus a `build` (K is a field, ≥ 1):
    - `MyGlobalGammaScreen(K)`;
    - `MyLiquidGammaScreen(K, shortlist, window)`;
    - `MyCorrGammaScreen(K, corr_max, window)`;
    - `MyGlobalRandomScreen(K)`;
    - `MyLiquidRandomScreen(K, shortlist, window)`;
    - `MyFixedScreen(tickers::Vector{String})`.
  - `pick(screen, md, t; X, γ_t, N, seed)::Vector{Int}`, the existing signature. The new screens ignore `N` (per-sector), use their own K, and return sorted columns, `min(K, eligible)` names. Rules (addendum §4):
    - **global γ:** the K highest `γ_t`; ties by index.
    - **liquid γ:** the K highest `γ_t` within `liquid_shortlist(md, t; size=shortlist, window)`.
    - **corr γ:** walk the global `γ_t` ranking descending.
      - Accept j if `cor(gⱼ, gₐ) ≤ corr_max` for every accepted a, where g is daily growth `log(close[k]/close[k−1])/Δt` for k ∈ (t−window+1):t.
      - A NaN correlation counts as not exceeding `corr_max`.
      - If fewer than K are accepted, fill with the highest-γ names not yet accepted.
    - **random global:** K distinct columns from `MersenneTwister(_stream_seed(seed, md.dates[t], 0))`.
    - **random liquid:** the same rule, over `liquid_shortlist(...)`.
    - **fixed:** the sorted columns of `tickers`. `ArgumentError` naming any ticker not in `md.tickers`.

- [ ] **Step 1: Write the failing tests**

```julia
using Test, Dates, Random, Statistics, VLINFORMS2026
include("helpers.jl")
# Market with constant closes (zero growth) except where `G` (days−1 × names, yr⁻¹) says otherwise,
# and volume (days × names, shares/day); 300 weekdays from 2020-01-01, one sector.
function uni_md(; n=300, m=48, G=zeros(n - 1, m), volume=ones(n, m))
    dates = first(filter(d -> dayofweek(d) <= 5, collect(Date(2020,1,1):Day(1):Date(2030,12,31))), n)
    cl = 10 .* exp.(vcat(zeros(1, m), cumsum(G ./ 252; dims=1)))
    build(MyMarketData, (dates=dates, tickers=["N$(lpad(j, 2, '0'))" for j in 1:m], open=cl, close=cl, volume=volume,
        spy_open=fill(100.0, n), spy_close=100 .* exp.(cumsum(vcat(0.0, 0.5 .* sin.(1:n-1)) ./ 252)), sector=fill("S1", m), g_f=zeros(n)))
end
X0 = zeros(1, 1)
@testset "global γ and liquid shortlist" begin
    md = uni_md(volume=repeat(Float64.((1:48)'), 300, 1))                     # dollar volume ∝ column index
    γ = collect(1:48) ./ 100
    @test pick(MyGlobalGammaScreen(5), md, 280; X=X0, γ_t=γ, N=3, seed=1) == [44, 45, 46, 47, 48]
    @test liquid_shortlist(md, 280; size=10, window=63) == collect(48:-1:39)
    v = repeat(Float64.((1:48)'), 300, 1); v[270, 1] = 1e12           # one huge day inside the window: median unmoved
    @test liquid_shortlist(uni_md(volume=v), 280; size=10, window=63) == collect(48:-1:39)
    γr = collect(48:-1:1) ./ 100                                     # name 1 has the highest γ, but it is illiquid
    @test pick(MyLiquidGammaScreen(3, 10, 63), md, 280; X=X0, γ_t=γr, N=3, seed=1) == [39, 40, 41]
    @test pick(MyGlobalGammaScreen(100), md, 280; X=X0, γ_t=γ, N=3, seed=1) == collect(1:48)   # K > universe
end
@testset "correlation-aware γ" begin                                # Review Focus 1, 3
    rng = MersenneTwister(3); G = randn(rng, 299, 6)
    G[:, 2] = G[:, 1]                                                # names 1 and 2 identical (ρ = 1)
    md = uni_md(m=6, G=G)
    γ = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    @test pick(MyCorrGammaScreen(2, 0.7, 252), md, 299; X=X0, γ_t=γ, N=3, seed=1) == [1, 3]
    Gs = repeat(G[:, 1], 1, 6); same = uni_md(m=6, G=Gs)             # every pair ρ = 1: one accepted, fill by γ
    @test pick(MyCorrGammaScreen(3, 0.7, 252), same, 299; X=X0, γ_t=γ, N=3, seed=1) == [1, 2, 3]
    Gz = copy(G); Gz[:, 3] .= 0.0                                    # name 3 has zero variance: NaN ρ is accepted
    @test 3 in pick(MyCorrGammaScreen(3, 0.7, 252), uni_md(m=6, G=Gz), 299; X=X0, γ_t=γ, N=3, seed=1)
    p = perturb_after(md, md.dates[250])                             # canary: data after t unused
    @test pick(MyCorrGammaScreen(2, 0.7, 200), p, 250; X=X0, γ_t=γ, N=3, seed=1) ==
          pick(MyCorrGammaScreen(2, 0.7, 200), md, 250; X=X0, γ_t=γ, N=3, seed=1)
end
@testset "random controls and the fixed list" begin                 # Review Focus 2
    md = uni_md(volume=repeat(Float64.((1:48)'), 300, 1)); γ = zeros(48)
    r = pick(MyGlobalRandomScreen(8), md, 280; X=X0, γ_t=γ, N=3, seed=11)
    @test length(r) == 8 && allunique(r) && issorted(r)
    @test r == pick(MyGlobalRandomScreen(8), md, 280; X=X0, γ_t=γ, N=3, seed=11)
    @test r != pick(MyGlobalRandomScreen(8), md, 280; X=X0, γ_t=γ, N=3, seed=12)
    rl = pick(MyLiquidRandomScreen(5, 10, 63), md, 280; X=X0, γ_t=γ, N=3, seed=11)
    @test length(rl) == 5 && all(in(39:48), rl)
    p = perturb_after(md, md.dates[280])
    @test pick(MyLiquidRandomScreen(5, 10, 63), p, 280; X=X0, γ_t=γ, N=3, seed=11) == rl
    @test pick(MyFixedScreen(["N05", "N02", "N40"]), md, 280; X=X0, γ_t=γ, N=3, seed=1) == [2, 5, 40]
    @test_throws ArgumentError pick(MyFixedScreen(["N05", "ZZZ"]), md, 280; X=X0, γ_t=γ, N=3, seed=1)
end
```

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_screens_universe.jl`. Expected: FAIL (types not defined).
- [ ] **Step 3: Implement** in `Screens.jl` and export the new names. Add the test file to `runtests.jl`.
- [ ] **Step 4: Run** the new file and `test_screens.jl`. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(screens): global, liquid and correlation-aware γ screens; random controls; fixed list"`

---

### Task 3: Settings, config, and the backtest's scaled pick cap

**Files:**
- Modify:
  - `code/src/StudySettings.jl` (fields, defaults, validation, `load_study` grid)
  - `config/study.toml` (new keys in the matching tables, new `[grid]` keys)
  - `code/src/StudyBacktest.jl` (`_STUDY_SCREENS`, `make_screen`, `_study_problem`, `run_study`)
- Test: append to `code/test/test_study_config.jl` and `code/test/test_study_backtest.jl`

**Interfaces:**
- Consumes: Task 1 (`w_cap`), Task 2 (screen types).
- Produces:
  - **`MyStudySettings` new fields and defaults:**
    - `basket_K::Int = 13`
    - `liquidity_shortlist::Int = 100`
    - `liquidity_window::Int = 63`
    - `corr_max::Float64 = 0.7`
    - `corr_window::Int = 252`
    - `w_max_scale::Float64 = 1.5`
    - `fixed_tickers::Vector{String} =` the user's 13-name list, in the order given in Global Constraints
  - **New validation:** `0 < corr_max ≤ 1`, `w_max_scale > 0`, `fixed_tickers` non-empty and unique. `context_key` is unchanged.
  - **`load_study(path).grid` new fields** (all required in `[grid]`):
    - `K::Vector{Int}`
    - `universe_screens::Vector{Symbol}`
    - `fixed_lists::Vector{Pair{String,Vector{String}}}`, read from the `[grid.fixed_lists]` table and sorted by name. Entries: `l8b_13` = the 13-name list, `l8b_15` = the 13 plus WMT, PG.
  - **`config/study.toml`:**
    - holds those values;
    - its `[grid]` gains `K = [5, 8, 13, 21, 34]` and `universe_screens = ["global_gamma", "liquid_gamma", "corr_gamma", "random_global", "random_liquid"]`;
    - the `[grid.fixed_lists]` table has the two lists.
  - **Screens:**
    - `_STUDY_SCREENS` adds `:global_gamma, :liquid_gamma, :corr_gamma, :random_global, :random_liquid, :fixed`.
    - `make_screen` builds them from `basket_K`, `liquidity_shortlist`, `liquidity_window`, `corr_max`, `corr_window` and `fixed_tickers`.
    - `const _STUDY_DETERMINISTIC = (:gamma, :global_gamma, :liquid_gamma, :corr_gamma, :fixed)`: these screens take a single seed in run lists.
  - **`_study_problem(ctx, run, t, cols, hedge, held, V; basket::Vector{Int}=Int[])`:**
    - the constraint set gets `w_cap[i] = (cols[i] in basket) ? max(st.w_max, st.w_max_scale/length(basket)) : st.w_max`;
    - with the default empty `basket`, every cap is `st.w_max`.
  - **`run_study`** passes the current screen basket at each decision. In the kept-book failure path the caps are irrelevant, so no basket is passed.

- [ ] **Step 1: Write the failing tests**

Append to `test_study_config.jl`:

```julia
@testset "addendum settings and grid" begin
    s = MyStudySettings()
    @test (s.basket_K, s.liquidity_shortlist, s.liquidity_window, s.corr_max, s.corr_window, s.w_max_scale) == (13, 100, 63, 0.7, 252, 1.5)
    @test s.fixed_tickers == ["AAPL", "MSFT", "INTC", "MU", "AMD", "NVDA", "GS", "BAC", "WFC", "C", "F", "GM", "JNJ"]
    cfg = load_study(joinpath(REPO, "config", "study.toml"))
    @test same(cfg.settings, s)
    @test cfg.grid.K == [5, 8, 13, 21, 34] && length(cfg.grid.universe_screens) == 5
    @test first.(cfg.grid.fixed_lists) == ["l8b_13", "l8b_15"] && last(cfg.grid.fixed_lists[2])[14:15] == ["WMT", "PG"]
    @test_throws ArgumentError build(MyStudySettings, (corr_max=1.5,))
    @test_throws ArgumentError build(MyStudySettings, (fixed_tickers=String[],))
end
```

Append to `test_study_backtest.jl`:

```julia
@testset "scaled cap for screen picks only" begin                     # Review Focus 4
    md, ctx = study_ctx(); run = study_run(ctx, "CD", :gamma); t = first(ctx.rows)
    cols = [1, 2, 3, 4, 5, 6]; hedge = [false, false, false, true, true, false]
    p = VLINFORMS2026._study_problem(ctx, run, t, cols, hedge, zeros(Int, 48), 1e5; basket=[1, 2, 3, 4, 6])
    @test p.constraints.w_cap ≈ [0.3, 0.3, 0.3, 0.3, 0.10, 0.3]        # K = 5 picks; pick 4 is also pooled; 5 is pool-only
    q = VLINFORMS2026._study_problem(ctx, run, t, cols, hedge, zeros(Int, 48), 1e5; basket=collect(1:33))
    @test all(==(0.10), q.constraints.w_cap)                         # 1.5/33 < 0.10
end
@testset "new screens and the fixed list run end to end" begin
    md, ctx = study_ctx(basket_K=8, fixed_tickers=md_tickers_13())
    for scr in (:global_gamma, :liquid_gamma, :corr_gamma, :random_global, :random_liquid, :fixed)
        res = run_study(ctx, study_run(ctx, "CD", scr))
        @test all(res.cash .>= 0) && all(isfinite, res.equity)
        scr == :fixed ? (@test !any(res.repicks.changed[2:end])) : (@test all(length.(res.repicks.basket) .== 8))
    end
end
```

Add the helper `md_tickers_13() = study_md().tickers[1:13]` to `helpers.jl` (the fixture is deterministic, so these are the context's tickers).

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_study_config.jl` and `test_study_backtest.jl`. Expected: FAIL (unknown fields).
- [ ] **Step 3: Implement** the Interfaces above.
- [ ] **Step 4: Run** both files, plus `test_study_runner.jl`, whose run counts must be unchanged. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(study): addendum settings and config; scaled per-stock cap for screen picks"`

---

### Task 4: Run lists, stats and tables for the size sweep

**Files:**
- Modify: `code/src/StudyRunner.jl`, `code/src/StudyTables.jl`
- Test: append to `code/test/test_study_runner.jl` and `code/test/test_study_tables.jl`

**Interfaces:**
- Consumes: Task 3 (settings, grid, screens).
- Produces:
  - **`universe_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}`:**
    - iterates `grid.K` × `grid.hedge` × `grid.allocators` × `grid.universe_screens`, with settings `with_settings(s; basket_K=K, hedge=h)`;
    - seeds `[grid.base_seed]` for `_STUDY_DETERMINISTIC` screens, else `base_seed .+ (0:n_seeds-1)`;
    - labels look like `"CD|global_gamma|K=13|hedge=true|seed=5660"`.
  - **`fixed_list_runs(s, grid)::Vector{MyStudyRun}`:**
    - iterates `grid.fixed_lists` × `grid.hedge` × `grid.allocators`, with screen `:fixed` and settings `with_settings(s; fixed_tickers=list, basket_K=length(list), hedge=h)`;
    - seed `base_seed`;
    - labels look like `"CD|fixed:l8b_13|hedge=true|seed=5660"`.
  - **`study_stats` adds `basket_size`:** the median of `length.(res.repicks.basket)`, a count.
  - **Summary CSV:** `rebuild_study_summary` writes `fixed_tickers` as `join(v, ";")`.
  - **`gamma_percentile` gains keywords `screen::Symbol=:gamma, control::Symbol=:random`.** The old defaults and behavior are unchanged, and the output column `gamma_value` keeps its name.
  - **`compare_summaries(old::DataFrame, new::DataFrame; key::Symbol=:label, cols=[:sharpe, :max_drawdown, :alpha, :alpha_t, :beta, :realized_growth, :terminal, :vol])::NamedTuple{(:n_matched, :n_missing, :max_abs_diff)}`:**
    - joins on `key`;
    - `max_abs_diff` is the largest |old − new| over `cols` and matched rows, treating NaN == NaN as 0;
    - `n_missing` counts keys in `old` that are absent from `new`.

- [ ] **Step 1: Write the failing tests**

Append to `test_study_runner.jl`, and add `Statistics` to the file's top-level `using` line (it needs `median`):

```julia
@testset "addendum run lists" begin
    s = MyStudySettings()
    g = (n_seeds=30, base_seed=5660, allocators=["CD", "CES", "SG", "MinVar", "Tangent", "LS"], screens=[:bandit, :gamma, :random],
        hedge=[true, false], K=[5, 8, 13, 21, 34],
        universe_screens=[:global_gamma, :liquid_gamma, :corr_gamma, :random_global, :random_liquid],
        fixed_lists=["l8b_13" => s.fixed_tickers, "l8b_15" => vcat(s.fixed_tickers, ["WMT", "PG"])])
    u = universe_runs(s, g); f = fixed_list_runs(s, g)
    @test length(u) == 3780 && length(f) == 24                      # 3·5·6·2 + 2·5·30·6·2; 2·6·2
    @test length(unique(study_hash.(vcat(u, f, tier1_runs(s, g))))) == 3780 + 24 + 732
    @test all(r -> r.settings.basket_K == 15, filter(r -> occursin("l8b_15", r.label), f))
    md, ctx = study_ctx(); res = run_study(ctx, study_run(ctx, "CD", :gamma))
    @test study_stats(res).basket_size == median(length.(res.repicks.basket))
end
```

Append to `test_study_tables.jl`:

```julia
@testset "matched-control percentile and summary comparison" begin
    df = DataFrame(allocator=fill("CD", 4), screen=["global_gamma", "random_global", "random_global", "random_global"],
        hedge=trues(4), basket_size=fill(13, 4), seed=[5660, 1, 2, 3], sharpe=[0.5, 0.2, 0.6, 0.4])
    p = gamma_percentile(df; screen=:global_gamma, control=:random_global, by=[:allocator, :basket_size, :hedge])
    @test only(p.percentile) ≈ 2 / 3 && only(p.gamma_value) == 0.5
    old = DataFrame(label=["a", "b"], sharpe=[0.1, NaN]); new = DataFrame(label=["a", "b"], sharpe=[0.1, NaN])
    @test compare_summaries(old, new; cols=[:sharpe]) == (n_matched=2, n_missing=0, max_abs_diff=0.0)
end
```

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_study_runner.jl` and `test_study_tables.jl`. Expected: FAIL.
- [ ] **Step 3: Implement** the Interfaces above and export the new names.
- [ ] **Step 4: Run** both files, then the full suite once. Expected: all PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(study): universe and fixed-list run lists, basket_size stat, matched percentile, summary comparison"`

---

### Task 5: Regression rerun, the size sweep, tables and figures

**Files:**
- Modify:
  - `scripts/11_run_study.jl` (add `--universe`: `vcat(universe_runs(s, grid), fixed_list_runs(s, grid))`, experiment `"universe"`)
  - `scripts/12_tables.jl` (new tables below)
  - `scripts/13_figures.jl` (figs 6 and 7)
  - `README.md` (run order and the addendum's plan decisions)
- Create: `scripts/15_regression_check.jl` (constants + `main()` only)
- Outputs (committed): `results/tier1_summary.csv`, `results/fixed_summary.csv`, `results/universe_summary.csv`, the new `results/tables/*.csv` and `results/figures/fig6_*.png`, `fig7_*.png`.

**Interfaces:**
- Consumes: Tasks 3–4.
- New tables written by `12_tables.jl`:
  - `size_sweep.csv`: `group_summary(universe ∪ tier1; by=[:allocator, :screen, :basket_size, :hedge])`.
  - `matched_percentile.csv`: `gamma_percentile` for (global_gamma vs random_global), (corr_gamma vs random_global) and (liquid_gamma vs random_liquid), by `[:allocator, :basket_size, :hedge]`, with a `pair` column naming the comparison.
  - `head_to_head_K13.csv`: `group_summary` over rows with `basket_size ∈ (13, 15)` (fixed lists and every screen and control), by `[:allocator, :screen, :basket_size, :hedge]`.
  - `quota_cost.csv`: medians for sector γ (tier1 `:gamma`, 33) vs `:global_gamma` at K = 34, and sector `:random` vs `:random_global` at K = 34, by allocator and hedge.
- New figures written by `13_figures.jl` (load the `dataviz` skill before writing plotting code):
  - `fig6_sharpe_vs_K.png` (1600×900): median Sharpe vs K, one line per deterministic screen. Random controls are shown as a median line with an IQR band. Horizontal lines for SPY and the equal-weight universe. Markers for the fixed lists at K = 13 and 15. One panel per allocator (CD, CES, SG, MinVar, Tangent, LS), hedge on.
  - `fig7_max_drawdown_vs_K.png`: the same for max drawdown.
  - Both captions say the new screens are exploratory, and that the user's lists were chosen in advance without 2025 data.

- [ ] **Step 1:** Rerun the base grids: `julia -t auto scripts/11_run_study.jl` then `julia -t auto scripts/11_run_study.jl --fixed`. Settings fields changed, so every hash is new and all runs execute. Expected: 732 and 366 runs, 0 errors.
- [ ] **Step 2:** Write `scripts/15_regression_check.jl`.
  - It loads the committed summaries with `git show HEAD:results/tier1_summary.csv` (and the fixed one) into DataFrames, plus the regenerated ones.
  - It prints `compare_summaries` for each and exits 1 unless `n_missing == 0 && max_abs_diff == 0.0`.
  - Run it. Expected: both pass, with zero difference. If not, stop and investigate as a bug (TDD fix in the owning test file); do not proceed.
- [ ] **Step 3:** Run `julia -t auto scripts/11_run_study.jl --universe`. Expected: 3,804 runs, 0 errors, and `results/universe_summary.csv` with 3,804 rows.
- [ ] **Step 4:** Extend and run `julia scripts/12_tables.jl`. Expected:
  - the four new CSVs are written;
  - `matched_percentile.csv` has 30 control seeds per row;
  - `head_to_head_K13.csv` contains `:fixed` rows at basket_size 13 and 15.
- [ ] **Step 5:** Extend and run `julia --project=scripts scripts/13_figures.jl`. Inspect fig6 and fig7 with the Read tool and fix any overlap or unreadable labels.
- [ ] **Step 6:** Update the README:
  - add `11 --universe` and `15` to the run order;
  - add one sentence each for the addendum's (my call) decisions: 63-day median dollar volume, top 100, ρ ≤ 0.7, cap factor 1.5, a pick-and-pool name gets the pick's cap, shorts unchanged, no fixed-schedule variant for the new screens;
  - add a sentence that the user's lists were chosen in advance without 2025 data.
- [ ] **Step 7: Commit** `git add scripts README.md results/*_summary.csv results/tables results/figures && git commit -m "results: ticker-picker addendum (size sweep, fixed lists) and base regression rerun"`
