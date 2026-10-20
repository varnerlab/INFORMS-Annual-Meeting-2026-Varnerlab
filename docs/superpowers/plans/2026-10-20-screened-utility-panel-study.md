# Screened Utility-Panel Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay the live trial's constrained-allocator/MPC machinery on 2025-01-02 → 2026-09-04 end-of-day data for six allocators × three re-picking ticker screens × hedge on/off, and produce growth-based scorecards, comparison tables and figures for the Nov 3 INFORMS talk.

**Architecture:**
- Extend the existing `VLINFORMS2026` package in `code/` (branch `feat/eod-study`, 754 tests) rather than rebuild it. Reuse its allocators, constraint set, GBM projection, costs, triggers, EWLS, SIM fit, covariance, λ choice, fill helpers and experiment-runner loop.
- Everything that depends only on market data is computed once per settings group in a **study context**: online SIM path, ξ and g̃_M, γ for the whole universe, the hedge pool for each day, and the bandit's excess-growth windows. A new day-by-day loop, `run_study`, then replays one run (allocator, screen, seed, hedge flag) against that context. It re-picks the basket inside the loop, because band exits move the re-pick dates.
- Every setting comes from `config/study.toml`. Numbered scripts calibrate on 2014–2024, run Tier 1, and write the tables and figures.

**Tech Stack:** Julia 1.12; JuMP + Clarabel (SCS fallback); DataFrames, CSV, JLD2, TOML, SHA, Statistics, LinearAlgebra, Random, Dates; Plots.jl for figures (scripts environment only); Test.

**Spec:** `docs/superpowers/specs/2026-10-20-screened-utility-panel-study-design.md`. Read it first; § numbers below refer to it.

**Branch:** create `feat/screened-panel` from `feat/eod-study` before Task 1. All commits go there.

**Schedule (Tier 1 results for slides by Fri Oct 30):**

| Tasks | Due |
|---|---|
| 1–7 | Thu Oct 22 |
| 8–11 | Sat Oct 24 |
| 12–13 | Sun Oct 25 |
| 14 (compute, tables, figures) | Mon Oct 26 – Wed Oct 28 |

## Global Constraints

- **Growth rates only (user, 2026-10-20):** the growth rate over k trading days is g = ln(S_end/S_start)/(k·Δt), in 1/yr, with Δt = 1/252. Nothing is computed as a simple or log return: not the bandit reward, not Sharpe, not alpha, not any label, field name, column or table heading. New code never uses the word "return" for a quantity.
- Compute every daily growth rate as `log(a / b) / Δt` with `Δt = 1/252` (the expression `growth_rates` uses). Replays and tests then match bit for bit.
- **Conventions (CHEME-5660):**
  - Every type is a `mutable struct My…` with `build(::Type{T}, data::NamedTuple)`. A type with defaults gets an inner zero-argument constructor that fills them, and its `build` starts from it.
  - Every function has a docstring that states units.
  - Seeded RNGs only; the default seed is 5660.
  - New source files are appended to `code/src/Include.jl` after `Experiments.jl` in this order: `MarketInputs.jl, Calibration.jl, OnlineSIM.jl, HedgePool.jl, Screens.jl, StudySettings.jl, Context.jl, GrowthScorecard.jl, StudyBacktest.jl, StudyRunner.jl, StudyTables.jl`. Exports go in the `export` block.
- **Settings:** every window, lag, limit and cost comes from `MyStudySettings` (Task 8), loaded from `config/study.toml`. The only literals allowed in new code are solver tolerances and the 1e-3-share floor ε already used by the trial engine.
- **No look-ahead:**
  - A decision after day t's close uses data dated ≤ t only. Orders fill at the open of t+1.
  - The bandit's reward windows must end on or before day t.
  - Every new module that touches prices has a perturb-after-d canary test (`perturb_after` in `code/test/helpers.jl`).
- **Windows:** calibration 2014-01-03 → 2024-12-31; run 2025-01-02 → 2026-09-04. Market data is loaded with `load_market_data(dir; start=Date(2014,1,3), stop=Date(2026,9,4))`.
- **Old code stays working:** `run_backtest`, `Validation.jl`, `Bandit.jl`, `scorecard` and their tests keep passing. New code may call their internal helpers (`_bt_fill!`, `_bt_adv`, `_bt_holdings_value`, `_bt_short_notional`, `_bt_projection_seed`, `_BTTrades`, `_sector_groups`, `_sample_without_replacement`, `_stream_seed`, `ema_series`).
- **Data:**
  - `data/`, `results/runs/` and `results/smoke/` are never committed.
  - Summary CSVs, tables, figures and `results/calibration*.{jld2,toml}` are committed.
- **Tests:**
  - Each new test file is added to `code/test/runtests.jl`.
  - Run one file with `cd code/test && julia --project=.. test_<name>.jl`; the full suite with `julia --project=code code/test/runtests.jl`.
- **Labels:** choices this plan makes that the spec leaves open are marked **(plan decision)** in docstrings and the README.

## Review Focus

1. **Stone–Geary baselines alone exceed β_max** (bearish ξ, many positive-β names at 1% each). Expected: `:infeasible`, previous holdings kept, the run continues without throwing. Test in Task 4.
2. **Hedge on, empty pool, binding cap** (all of Jan–Mar 2025). Expected: the γ-based rules trim or hold cash, Σwβ ≤ β_max holds, the status stays `:optimal`. Test in Task 4 (all-positive-β CD at β_max = 0.1); Task 10 also asserts the cap at every decision of every allocator.
3. **A pool name the screen also picked.** Expected: it appears once in the allocation problem and is flagged as a hedge. Test in Task 10 (`_study_problem_columns`).
4. **A 130/30 short candidate whose γ is ≥ 0 at a later decision, and pool names in the 130/30.** Expected: the short candidate is left out of the problem (target 0, never shorted), and pool names are never short. Tests in Tasks 4 and 10.
5. **A sector with fewer names than N, or fewer γ < 0 names than N.** Expected: the screen or short list takes all the sector has; no error. Test in Task 7.

---

### Task 1: EWLS market prior and training market moments

**Files:**
- Modify: `code/src/SIM.jl` (`ewls_init`, `fit_base_sim`)
- Test: `code/test/test_sim.jl` (append)

**Interfaces:**
- Produces:
  - `ewls_init(α₀, β₀, σ_ε₀; half_life=63.0, prior_weight=63.0, market_mean=0.0, market_variance=1.0)::MyEWLSState`. The defaults reproduce today's legacy statistics exactly.
  - `fit_base_sim(md, start, stop)` returns `(params, σ_m, m_m, v_m)`:
    - `m_m` = mean SPY growth (yr⁻¹).
    - `v_m` = population variance of SPY growth (`var(g; corrected=false)`, yr⁻²).
    - Same days as `σ_m`.

- [ ] **Step 1: Write the failing tests** (append to `test_sim.jl`, inside a new testset)

```julia
@testset "ewls_init market prior (CHEME-5660 seeding)" begin
    m, v = 0.12, 7.0
    s = ewls_init(0.05, 1.0, 2.0; market_mean=m, market_variance=v)
    @test s.Swx ≈ 63m && s.Swxx ≈ 63(v + m^2)
    β = (s.Sw*s.Swxy - s.Swx*s.Swy) / (s.Sw*s.Swxx - s.Swx^2); α = (s.Swy - β*s.Swx) / s.Sw
    @test β ≈ 1.0 rtol=1e-12
    @test α ≈ 0.05 rtol=1e-10
    l = ewls_init(0.05, 1.0, 2.0)                                  # legacy default unchanged
    @test (l.Swx, l.Swxx, l.Swxy, l.Swyy) == (0.0, 63.0, 63.0, 63 * (0.05^2 + 1.0 + 4.0))
    # 64 data points with the prior's market moments pull β exactly halfway to the data's β = 0;
    # the legacy unit-variance seed is ~7× weaker in growth units and lets β fall below 0.2.
    x = [isodd(k) ? m + sqrt(v) : m - sqrt(v) for k in 1:64]
    p = ewls_init(0.0, 1.0, 1.0; half_life=Inf, prior_weight=64.0, market_mean=m, market_variance=v)
    q = ewls_init(0.0, 1.0, 1.0; half_life=Inf, prior_weight=64.0)
    foreach(xk -> (ewls_update!(p, 0.0, xk); ewls_update!(q, 0.0, xk)), x)
    @test p.β ≈ 0.5 rtol=1e-9
    @test q.β < 0.2
end

@testset "fit_base_sim market moments" begin
    md = synthetic_md(ρ=0.3)
    fit = fit_base_sim(md, md.dates[1], md.dates[800])
    g = growth_rates(md.spy_close[1:800])
    @test fit.m_m ≈ mean(g) && fit.v_m ≈ var(g; corrected=false) && fit.σ_m ≈ std(g)
end
```

(`test_sim.jl` needs `include("helpers.jl")` and `using Statistics` if not already present.)

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_sim.jl`. Expected: FAIL (`market_mean` keyword unknown; `m_m` missing).

- [ ] **Step 3: Implement.** In `ewls_init`, use the CHEME-5660 `AdaptivePortfolio.jl:713` formulas with m = `market_mean`, v = `market_variance`, Exx = v + m², N₀ = `prior_weight`: Swx = N₀m, Swxx = N₀·Exx, Swy = N₀(α₀ + β₀m), Swxy = N₀(α₀m + β₀·Exx), Swyy = N₀(α₀² + 2α₀β₀m + β₀²·Exx + σ_ε₀²). Validate as the course does (finite, positive variance and weight). `half_life = Inf` must give η = 1. In `fit_base_sim`, add `m_m` and `v_m` to the returned NamedTuple.

- [ ] **Step 4: Run** the same command; then the full suite. Expected: PASS, with the old EWLS and backtest tests unchanged.

- [ ] **Step 5: Commit** `git commit -am "feat(sim): EWLS market-moment prior and training market moments"`

---

### Task 2: Market inputs ξ, g̃_M and the signed γ

**Files:**
- Create: `code/src/MarketInputs.jl`
- Test: `code/test/test_market_inputs.jl`

**Interfaces:**
- Consumes: `ema_series(prices; span)` (SIM.jl), `growth_rates`.
- Produces:
  - `xi_gain(spy_close::AbstractVector; short::Int=21, long::Int=63, q::Float64=0.99)::Float64`. L8b's gain, from the training closes only: G = 1/quantile(|EMA_short/EMA_long − 1|[t₀:end], q), with t₀ = short + long and EMAs seeded at the first close.
  - `xi_series(spy_close::AbstractVector; short::Int=21, long::Int=63, G::Float64)::Vector{Float64}`. ξ[t] = −G·(EMA_short[t]/EMA_long[t] − 1) on the full series; causal.
  - `gtilde_series(spy_close::AbstractVector; window::Int=252, Δt::Float64=1/252)::Vector{Float64}`. Element 1 is `NaN`; element k ≥ 2 is the span-`window` EMA (seeded at the first growth) of SPY daily growth through day k (yr⁻¹).
  - `gamma_weights(α::AbstractVector, β::AbstractVector, g̃_M::Real, ξ::Real; β_floor::Real=0.1)::Vector{Float64}`. γᵢ = tanh((αᵢ + βᵢ·g̃_M) / max(|βᵢ|, β_floor)^ξ) (§5 step 3).

- [ ] **Step 1: Write the failing tests**

```julia
using Test, Statistics, VLINFORMS2026
const ema = VLINFORMS2026.ema_series
@testset "market inputs" begin
    k = 1:600
    spy = 100 .* exp.(cumsum(0.01 .* sin.(0.05 .* k) .+ 0.0004))
    G = xi_gain(spy; short=21, long=63, q=0.99)
    r = abs.(ema(spy; span=21) ./ ema(spy; span=63) .- 1)[84:end]
    @test G ≈ 1 / quantile(r, 0.99)
    ξ = xi_series(spy; short=21, long=63, G=G)
    @test ξ ≈ -G .* (ema(spy; span=21) ./ ema(spy; span=63) .- 1)
    @test mean(abs.(ξ[84:end]) .<= 1 + 1e-12) >= 0.99
    g̃ = gtilde_series(spy; window=252)
    @test isnan(g̃[1]) && g̃[2:end] ≈ ema(growth_rates(spy); span=252)
    c = 100 .* exp.(0.15 .* (0:299) ./ 252)                         # constant growth 0.15 yr⁻¹
    @test all(isapprox.(gtilde_series(c)[2:end], 0.15; rtol=1e-9))
    # causal: a price change at day 400 leaves days 1..399 untouched
    p = copy(spy); p[400] *= 1.1
    @test xi_series(p; G=G)[1:399] == ξ[1:399] && xi_series(p; G=G)[400] != ξ[400]
    @test gtilde_series(p)[2:399] == g̃[2:399] && gtilde_series(p)[400] != g̃[400]
end

@testset "signed γ" begin
    α = [0.10, 0.05, 0.02, 0.03]; β = [1.3, 0.6, -0.5, 0.01]; g̃ = 0.15; ξ = 0.3
    γ = gamma_weights(α, β, g̃, ξ; β_floor=0.1)
    @test γ[1:2] ≈ tanh.(α[1:2] ./ β[1:2] .^ ξ .+ β[1:2] .^ (1 - ξ) .* g̃)   # L8b for β ≥ β_floor
    @test γ[3] ≈ tanh((0.02 - 0.5 * 0.15) / 0.5^0.3) && γ[3] < 0           # charged, not credited
    @test tanh(0.02 / 0.5^0.3 + 0.5^0.7 * 0.15) > 0                         # the trial's |β| form would prefer it
    @test γ[4] ≈ tanh((0.03 + 0.01 * 0.15) / 0.1^0.3)                       # β_floor
    @test all(isfinite, gamma_weights([0.1], [0.0], g̃, -0.7))
end
```

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_market_inputs.jl`. Expected: FAIL (`xi_gain` not defined).
- [ ] **Step 3: Implement** the four functions in `MarketInputs.jl`, add the include and exports, and add the test file to `runtests.jl`.
- [ ] **Step 4: Run** the test file. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: market inputs ξ, g̃_M and signed γ"` (with the new files added).

---

### Task 3: Calibration on 2014–2024

**Files:**
- Create: `code/src/Calibration.jl`
- Test: `code/test/test_calibration.jl`

**Interfaces:**
- Consumes: `fit_base_sim` (Task 1), `choose_shrinkage(md, start, stop; grid=0:0.1:1, n_portfolios=200, seed=5660)`.
- Produces:
  - `MyCalibration` with fields:
    - `base_sim::Dict{String,MySIMParameterEstimate}`
    - `σ_m` (yr⁻¹, N−1 SD of SPY growth)
    - `m_m` (yr⁻¹)
    - `v_m` (yr⁻², population)
    - `λ` (dimensionless)
    - `train_start::Date`, `train_stop::Date`
  - `build(MyCalibration, data)`.
  - `calibrate(md::MyMarketData; train_start::Date, train_stop::Date, seed::Int=5660, n_portfolios::Int=200)::MyCalibration`.
  - `save_calibration(path::AbstractString, c::MyCalibration)` and `load_calibration(path)::MyCalibration` (JLD2, key `"calibration"`).
  - Note: G is not stored here. It depends on the configured ξ windows and is computed in the context (Task 8). The vol cap σ_g,max defaults to `σ_m` (§4).

- [ ] **Step 1: Write the failing test**

```julia
using Test, Dates, Statistics, VLINFORMS2026
include("helpers.jl")
@testset "calibration" begin
    md = synthetic_md(ρ=0.3)
    a, b = md.dates[1], md.dates[1000]
    c = calibrate(md; train_start=a, train_stop=b, n_portfolios=50)
    fit = fit_base_sim(md, a, b)
    @test all(c.base_sim[t].β == fit.params[t].β && c.base_sim[t].α == fit.params[t].α for t in md.tickers)
    @test (c.σ_m, c.m_m, c.v_m) == (fit.σ_m, fit.m_m, fit.v_m)
    @test c.λ == choose_shrinkage(md, a, b; n_portfolios=50, seed=5660)
    @test (c.train_start, c.train_stop) == (a, b)
    path = joinpath(mktempdir(), "cal.jld2"); save_calibration(path, c); d = load_calibration(path)
    @test d.λ == c.λ && d.σ_m == c.σ_m && d.base_sim[md.tickers[3]].β == c.base_sim[md.tickers[3]].β
end
```

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_calibration.jl`. Expected: FAIL.
- [ ] **Step 3: Implement** `Calibration.jl`, the include, exports and the `runtests.jl` entry.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: 2014–2024 calibration (SIM fit, market moments, λ)"`

---

### Task 4: β cap, hedge names, and CES below one

**Files:**
- Modify:
  - `code/src/Allocators.jl` (`MyConstraintSet`, `MyAllocationProblem`, `build`s, `_check_problem`, `add_shared_constraints!`, `_solve_utility`, the CD/CES/SG `allocate`s)
  - `code/src/AllocatorsMeanVariance.jl` (`_without_vol_cap`)
  - `code/src/AllocatorsLongShort.jl`
  - `code/test/helpers.jl` (`P`)
  - `code/test/test_allocators_utility.jl` (replace the `η=1.0` throw test)
- Test: `code/test/test_allocators_hedge.jl`

**Interfaces:**
- Produces:
  - `MyConstraintSet` gains:
    - `β::Vector{Float64}`: each name's signed online β; optional in `build`, default `zeros(K)`, length must equal `sector`.
    - `β_max::Float64`: optional, default `Inf` (= off); `NaN` throws.
  - `MyAllocationProblem` gains `hedge::Vector{Bool}`: optional in `build`, default all `false`, length K.
  - `add_shared_constraints!` adds the row Σ βᵢwᵢ ≤ β_max·κ via `_add_le!` when `isfinite(β_max)`, for both long-only and `signed=true`. The β cap therefore reaches `max_investable`, the min-variance and tangent mixes, and the 130/30.
  - `_without_vol_cap` copies `β`, `β_max` and `hedge`.
  - Utility rules (CD, CES, SG):
    - A name with `hedge[i] && γᵢ ≤ 0` is a **free** variable. Its lower bound is the rule's pin (ε for CD/CES, bᵢ for SG). It gets no utility term.
    - Every other γ ≤ 0 name stays pinned, as today.
    - The objective subtracts `_HEDGE_TIEBREAK = 1e-6` × Σ wᵢ over the free hedge names **(plan decision)**. On the flat direction an interior-point solver otherwise returns an arbitrary hedge holding. With the tie-break, a hedge leaves its floor only when a binding limit makes it worth its budget.
    - Change `_solve_utility`'s contract: the `objective!(m, x, γp)` closure **returns** the utility expression, and `_solve_utility` sets `@objective(m, Max, U − tiebreak)`.
  - `MyCESRule`:
    - `η > 1`: unchanged.
    - `η == 1`: delegates to `MyCobbDouglasRule`.
    - `0 < η < 1`: minimizes Σ γᵢ tᵢ with `[tᵢ, nᵢ, 1.0] in MOI.PowerCone(η)`, i.e. tᵢ ≥ nᵢ^ρ with ρ = 1 − 1/η < 0. Budget-only shares are then ∝ (γ/S)^η, as in CHEME-5660 `allocate_ces`. Update the docstrings.
  - `MyLongShortCDRule`:
    - Long set = names with γ > 0 or `hedge`.
    - Short set = names with γ < 0 and not `hedge`.
    - Hedge names with γ ≤ 0 are long variables (lower bound 1e-8) with no utility term and the same tie-break.
    - No γ > 0 → `:no_preferred` (all cash), as today.
- Test helper: `P(...)` gains keyword arguments `β=nothing` (→ zeros), `β_max=Inf`, `hedge=nothing` (→ all false), passed into the two `build` calls.

- [ ] **Step 1: Write the failing tests** (`test_allocators_hedge.jl`)

```julia
using Test, LinearAlgebra, VLINFORMS2026
include("helpers.jl")
const RULES_LONG = (MyCobbDouglasRule(), build(MyCESRule, (η=0.5,)), build(MyCESRule, (η=2.0,)),
    build(MyStoneGearyRule, (baseline_frac=0.01,)))

@testset "CES below one and at one" begin
    γ = [0.2, 0.5, 0.3]; S = [10.0, 20.0, 40.0]
    @test allocate(build(MyCESRule, (η=0.5,)), P()).n ≈ 1000 .* (γ ./ S) .^ 0.5 ./ sum(S .* (γ ./ S) .^ 0.5) rtol=1e-5
    @test allocate(build(MyCESRule, (η=1.0,)), P()).n ≈ allocate(MyCobbDouglasRule(), P()).n rtol=1e-8
end

@testset "β cap binds in every rule" begin
    β = [1.5, 1.0, 0.5]
    for rule in (RULES_LONG..., MyMinVarRule(), MyTangentRule(), MyLongShortCDRule())
        r = allocate(rule, P(β=β, β_max=0.8, μ_g=[0.3, 0.2, 0.1]))
        @test r.status in (:optimal, :fallback_minvar, :long_only_fallback)
        @test dot(r.w, β) <= 0.8 + 1e-6
    end
    @test dot(allocate(MyCobbDouglasRule(), P(β=β, β_max=0.8)).w, β) ≈ 0.8 rtol=1e-5   # uncapped: 0.95
    @test allocate(MyCobbDouglasRule(), P(β=β, β_max=Inf)).n ≈ allocate(MyCobbDouglasRule(), P()).n
    r = allocate(MyCobbDouglasRule(), P(β=[1.0, 1.0, 1.0], β_max=0.1))     # Review Focus 2: cash, not infeasible
    @test r.status == :optimal && r.s_star <= 0.1 + 1e-6
end

@testset "hedge names: bought to meet the cap, at the floor otherwise" begin
    γ = [0.5, 0.4, -0.2, -0.1]; β = [1.2, 1.0, -0.5, -0.5]; S = [10.0, 20.0, 40.0, 25.0]
    hedge = [false, false, true, false]
    for rule in RULES_LONG
        sg = rule isa MyStoneGearyRule
        r = allocate(rule, P(γ=γ, prices=S, β=β, β_max=0.4, hedge=hedge, w_max=0.5))
        @test r.status == :optimal
        @test r.w[3] > 0.05                                       # the hedge leaves its floor
        @test r.n[4] == (sg ? 0.0 : 1e-3)                          # a non-pool γ ≤ 0 name stays pinned
        @test dot(r.w, β) <= 0.4 + 1e-6
        r0 = allocate(rule, P(γ=γ, prices=S, β=β, β_max=Inf, hedge=hedge, w_max=0.2))   # budget slack
        @test r0.n[3] <= (sg ? 0.0 : 1e-3) + 1e-2                  # tie-break keeps it at the floor
    end
end

@testset "130/30: pool names long, others short" begin
    γ = [0.5, 0.4, -0.3, -0.2]; β = [1.2, 1.0, -0.6, 1.1]
    r = allocate(MyLongShortCDRule(), P(γ=γ, prices=[10.0, 20.0, 40.0, 25.0], β=β, β_max=0.3,
        hedge=[false, false, true, false], w_max=0.6))
    @test r.status == :optimal && r.n[3] > 0 && r.n[4] < 0
    @test dot(r.w, β) <= 0.3 + 1e-6
end

@testset "Stone–Geary baselines above the cap" begin                  # Review Focus 1
    r = allocate(build(MyStoneGearyRule, (baseline_frac=0.3,)), P(β=[1.0, 1.0, 1.0], β_max=0.5, n_prev=[1.0, 2.0, 3.0]))
    @test r.status == :infeasible && r.n == [1.0, 2.0, 3.0]
end

@testset "constraint-set validation" begin
    @test_throws DimensionMismatch build(MyConstraintSet, (w_max=0.1, s_max=1.0, σ_g_max=1.0, K_turnover=1.0,
        c̄=0.0, sector=["A", "B"], n_prev=[0.0, 0.0], β=[1.0]))
    @test_throws ArgumentError build(MyConstraintSet, (w_max=0.1, s_max=1.0, σ_g_max=1.0, K_turnover=1.0,
        c̄=0.0, sector=["A"], n_prev=[0.0], β_max=NaN))
end
```

In `test_allocators_utility.jl`, replace `@test_throws ArgumentError allocate(build(MyCESRule,(η=1.0,)), P())` with `@test allocate(build(MyCESRule,(η=1.0,)), P()).n ≈ allocate(MyCobbDouglasRule(), P()).n rtol=1e-8`.

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_allocators_hedge.jl`. Expected: FAIL (`β` is not a `MyConstraintSet` field).
- [ ] **Step 3: Implement** the changes listed under Interfaces.
- [ ] **Step 4: Run** the new file, `test_allocators_utility.jl` and `test_allocators_mv_ls.jl`; then the full suite. Expected: all PASS, with the engine fixture test and the old LS tests unchanged, since every old problem has `hedge` all false and `β_max = Inf`.
- [ ] **Step 5: Commit** `git commit -m "feat(allocators): β cap, free hedge names, CES η ≤ 1"`

---

### Task 5: Online SIM path for the whole universe

**Files:**
- Create: `code/src/OnlineSIM.jl`
- Test: `code/test/test_online_sim.jl`

**Interfaces:**
- Consumes: `ewls_init(...; market_mean, market_variance)` (Task 1), `ewls_update!`, `ewvar_init`, `ewvar_update!`, `MyCalibration` (Task 3).
- Produces:
  - `MyOnlineSIM` with fields `rows::UnitRange{Int}` (md row indices of the run) and `α, β, σ_ε::Matrix{Float64}` (length(rows) × length(md.tickers); yr⁻¹, dimensionless, yr⁻¹).
  - `σ_m::Vector{Float64}` (yr⁻¹).
  - Row s holds the estimates **after** the update with the growth into day `rows[s]`.
  - `online_sim(md::MyMarketData, calib::MyCalibration; start::Date, stop::Date, half_life::Float64=63.0, prior_weight::Float64=63.0)::MyOnlineSIM`:
    - `rows = index_of(md, start):searchsortedlast(md.dates, stop)`.
    - `start` needs a previous trading day.
    - Every ticker must have a `calib.base_sim` entry (`ArgumentError` otherwise).
    - EWLS states are seeded with `market_mean = calib.m_m`, `market_variance = calib.v_m`.
    - σ_m follows `ewvar_init(calib.σ_m; half_life, prior_weight)` / `ewvar_update!` with SPY growth.

- [ ] **Step 1: Write the failing test**

```julia
using Test, Dates, VLINFORMS2026
include("helpers.jl")
function toy_calib(md, k)
    fit = fit_base_sim(md, md.dates[1], md.dates[k])
    build(MyCalibration, (base_sim=fit.params, σ_m=fit.σ_m, m_m=fit.m_m, v_m=fit.v_m, λ=0.5,
        train_start=md.dates[1], train_stop=md.dates[k]))
end
@testset "online SIM path" begin
    md = synthetic_md(ρ=0.3); c = toy_calib(md, 800)
    sim = online_sim(md, c; start=md.dates[801], stop=md.dates[900])
    @test sim.rows == 801:900 && size(sim.β) == (100, 48)
    j = 7; p = c.base_sim[md.tickers[j]]
    s = ewls_init(p.α, p.β, p.σ_ε; market_mean=c.m_m, market_variance=c.v_m)
    v = ewvar_init(c.σ_m); σ = 0.0
    for (k, i) in enumerate(801:900)
        g_m = log(md.spy_close[i] / md.spy_close[i-1]) / (1/252)
        ewls_update!(s, log(md.close[i, j] / md.close[i-1, j]) / (1/252), g_m); σ = ewvar_update!(v, g_m)
        k == 50 && @test (sim.α[k, j], sim.β[k, j], sim.σ_ε[k, j], sim.σ_m[k]) == (s.α, s.β, s.σ_ε, σ)
    end
    q = online_sim(perturb_after(md, md.dates[850]), c; start=md.dates[801], stop=md.dates[900])
    @test q.β[1:50, :] == sim.β[1:50, :] && q.β[51, :] != sim.β[51, :]          # canary
    @test_throws ArgumentError online_sim(md, c; start=md.dates[1], stop=md.dates[10])
end
```

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_online_sim.jl`. Expected: FAIL.
- [ ] **Step 3: Implement**, add the include, exports and the `runtests.jl` entry.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: online SIM path for the universe"`

---

### Task 6: Down-day β, hedge pool, β cap

**Files:**
- Create: `code/src/HedgePool.jl`
- Test: `code/test/test_hedge_pool.jl`

**Interfaces:**
- Produces:
  - `down_day_beta(md::MyMarketData, t::Int; lookback::Int=252)::Vector{Float64}`, one per ticker:
    - The OLS slope, with intercept, of the ticker's daily growth on SPY's daily growth.
    - Uses the days k ∈ max(2, t−lookback+1):t on which SPY's growth into k is < 0.
    - `NaN` when fewer than 3 such days.
  - `hedge_pool(β_t::AbstractVector, βdown_t::AbstractVector; H::Int=5)::Vector{Int}`:
    - Eligible columns have `β_t[j] ≤ 0` and `βdown_t[j] ≤ 0` (`NaN` is not eligible).
    - Ranked by `βdown_t` ascending, ties by column index; the first `H`.
  - `beta_cap(ξ::Real; β_ref::Real=1.0, κ_β::Real=1.0, β_min::Real=0.0)::Float64` = max(β_min, β_ref − κ_β·ξ).

- [ ] **Step 1: Write the failing tests**

```julia
using Test, Dates, VLINFORMS2026
include("helpers.jl")
# Market from SPY growth gm (yr⁻¹, one per day after the first) and ticker growth G (days−1 × names).
function growth_md(gm::Vector{Float64}, G::Matrix{Float64})
    n = length(gm) + 1; m = size(G, 2)
    dates = first(filter(d -> dayofweek(d) <= 5, collect(Date(2020,1,1):Day(1):Date(2030,12,31))), n)
    cl = 100 .* exp.(vcat(zeros(1, m), cumsum(G ./ 252; dims=1))); spy = 100 .* exp.(vcat(0.0, cumsum(gm ./ 252)))
    build(MyMarketData, (dates=dates, tickers=["N$(j)" for j in 1:m], open=cl, close=cl, volume=fill(1e6, n, m),
        spy_open=spy, spy_close=spy, sector=fill("S1", m), g_f=zeros(n)))
end
@testset "down-day β" begin
    gm = 2.0 .* sin.(0.7 .* (1:299))
    G = hcat([x < 0 ? -0.5x : 0.8x for x in gm], gm)                    # hedge on down days; plain β = 1
    md = growth_md(gm, G)
    b = down_day_beta(md, 300; lookback=252)
    @test b[1] ≈ -0.5 rtol=1e-8 && b[2] ≈ 1.0 rtol=1e-8
    G2 = copy(G); G2[10, 1] += 50.0                                     # outside the window: no effect
    @test down_day_beta(growth_md(gm, G2), 300; lookback=252) ≈ b rtol=1e-10   # later prices shift: last-bit noise only
    @test all(isnan, down_day_beta(growth_md(abs.(gm), G), 300; lookback=252))   # no down days
end
@testset "hedge pool and β cap" begin
    β_t = [-0.2, -0.1, 0.3, -0.4, -0.05, -0.3, -0.01]
    βd = [-0.3, 0.2, -0.5, -0.1, NaN, -0.3, -0.6]
    @test hedge_pool(β_t, βd; H=5) == [7, 1, 6, 4]
    @test hedge_pool(β_t, βd; H=2) == [7, 1]
    @test isempty(hedge_pool(fill(0.5, 3), fill(-1.0, 3)))
    @test beta_cap(0.0) == 1.0 && beta_cap(0.5) == 0.5 && beta_cap(-0.5) == 1.5 && beta_cap(2.0) == 0.0
    @test beta_cap(0.5; β_ref=1.2, κ_β=0.4, β_min=0.3) ≈ 1.0
end
```

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_hedge_pool.jl`. Expected: FAIL.
- [ ] **Step 3: Implement**, add the include, exports and the `runtests.jl` entry.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: down-day β, hedge pool and ξ-driven β cap"`

---

### Task 7: Ticker screens and 130/30 short candidates

**Files:**
- Create: `code/src/Screens.jl`
- Test: `code/test/test_screens.jl`

**Interfaces:**
- Consumes: `_sector_groups(md)`, `_sample_without_replacement(rng, pool, k)`, `_stream_seed(seed, date, s)` (Bandit.jl).
- Produces:
  - `abstract type AbstractScreen end`.
  - `MyBanditScreen <: AbstractScreen` with fields `lookback::Int, horizon::Int, ε_floor::Float64, iters_per_arm::Int, min_iters::Int, max_iters::Int`. Inner defaults: 252, 21, 0.05, 50, 500, 5000.
  - `MyGammaScreen`, `MyRandomScreen` (no fields). Each type has a `build`.
  - `excess_growth_windows(md::MyMarketData; horizon::Int=21, Δt::Float64=1/252)::Matrix{Float64}`. Size (length(md.dates) − horizon) × tickers, with X[d, j] = (log(close[d+h, j]/close[d, j]) − log(spy[d+h]/spy[d])) / (h·Δt), in yr⁻¹.
  - `train_ticker_bandit(arms::Vector{Int}, days::UnitRange{Int}, X::AbstractMatrix; seed::Int, ε_floor=0.05, iters_per_arm=50, min_iters=500, max_iters=5000)::NamedTuple{(:mean, :count, :tiebreak)}`. Vectors are aligned with `arms`. The algorithm, in this order:
    1. `rng = MersenneTwister(seed)`.
    2. `tiebreak = randperm(rng, A)`, with A = length(arms).
    3. iters = clamp(iters_per_arm·A, min_iters, max_iters).
    4. For k = 1..iters:
       - ε = max(ε_floor, k > 1 ? min(1, k^(−1/3)·(A·log k)^(1/3)) : 1).
       - If `rand(rng) < ε` or nothing has been pulled yet, take a uniform arm (`rand(rng, 1:A)`). Otherwise take the pulled arm with the highest mean, ties going to the smaller `tiebreak`.
       - Draw the day `d = rand(rng, days)` and the reward `X[d, arms[a]]`.
       - Update the running mean and count.
  - `rank_arms(res)::Vector{Int}`: positions ordered by pulled first, then mean descending, then `tiebreak` ascending.
  - `pick(screen::AbstractScreen, md::MyMarketData, t::Int; X::AbstractMatrix, γ_t::AbstractVector, N::Int, seed::Int)::Vector{Int}`. Returns sorted column indices, N per sector (all of a sector with ≤ N names). Sectors are taken in sorted-name order with rank s = 1, 2, ….
    - **Bandit:**
      - `size(X, 1) == length(md.dates) − screen.horizon`, else `ArgumentError`.
      - Training days are `(t − lookback + 1 − horizon):(t − horizon)`, so every window ends in (t − lookback, t]. If the range starts below 1, throw `ArgumentError`.
      - Seed per sector: `_stream_seed(seed, md.dates[t], s)`.
      - Keeps the first N of `rank_arms`.
    - **γ:** the N highest `γ_t` per sector, ties by column index. Ignores `X` and `seed`.
    - **Random:** one `MersenneTwister(_stream_seed(seed, md.dates[t], 0))` stream; `_sample_without_replacement` per sector, in sector order.
  - `short_candidates(γ_t::AbstractVector, md::MyMarketData; N::Int, exclude::AbstractVector{Int}=Int[])::Vector{Int}`. Per sector, the up-to-N names with γ < 0 not in `exclude`, most negative first; result sorted.

- [ ] **Step 1: Write the failing tests**

```julia
using Test, Dates, Random, VLINFORMS2026
include("helpers.jl")
@testset "screens" begin
    md = synthetic_md(ρ=0.3); X = excess_growth_windows(md; horizon=21)
    @test size(X) == (length(md.dates) - 21, 48)
    @test X[100, 5] ≈ (log(md.close[121, 5] / md.close[100, 5]) - log(md.spy_close[121] / md.spy_close[100])) / (21 / 252)

    rng = MersenneTwister(1); Xt = 0.1 .* randn(rng, 200, 10); Xt[:, 4] .+= 1.0
    res = train_ticker_bandit(collect(1:10), 1:200, Xt; seed=7)
    @test sum(res.count) == 500 && first(rank_arms(res)) == 4
    @test train_ticker_bandit(collect(1:10), 1:200, Xt; seed=7).mean == res.mean
    @test sum(train_ticker_bandit(collect(1:120), 1:200, 0.1 .* randn(rng, 200, 120); seed=7).count) == 5000

    t = 1000; γ = collect(1:48) ./ 100
    b = pick(MyBanditScreen(), md, t; X=X, γ_t=γ, N=3, seed=5660)
    @test length(b) == 12 && issorted(b) && all(count(j -> md.sector[j] == s, b) == 3 for s in unique(md.sector))
    X2 = copy(X); X2[(t - 21 + 1):end, :] .= 1e6                         # windows ending after t: unused
    @test pick(MyBanditScreen(), md, t; X=X2, γ_t=γ, N=3, seed=5660) == b
    p = perturb_after(md, md.dates[t])
    @test pick(MyBanditScreen(), p, t; X=excess_growth_windows(p), γ_t=γ, N=3, seed=5660) == b
    @test_throws ArgumentError pick(MyBanditScreen(), md, 100; X=X, γ_t=γ, N=3, seed=1)

    @test pick(MyGammaScreen(), md, t; X=X, γ_t=γ, N=3, seed=1) == [10, 11, 12, 22, 23, 24, 34, 35, 36, 46, 47, 48]
    r1 = pick(MyRandomScreen(), md, t; X=X, γ_t=γ, N=3, seed=1)
    @test r1 == pick(MyRandomScreen(), md, t; X=X, γ_t=γ, N=3, seed=1)
    @test r1 != pick(MyRandomScreen(), md, t; X=X, γ_t=γ, N=3, seed=2)
    @test r1 != pick(MyRandomScreen(), md, t + 1; X=X, γ_t=γ, N=3, seed=1)
end
@testset "small sectors and short candidates" begin                   # Review Focus 5
    md0 = synthetic_md(ρ=0.3)
    md = build(MyMarketData, merge(NamedTuple(f => getfield(md0, f) for f in fieldnames(MyMarketData)),
        (sector=vcat(md0.sector[1:46], ["S9", "S9"]),)))
    X = excess_growth_windows(md); γ = collect(1:48) ./ 100
    for scr in (MyBanditScreen(), MyGammaScreen(), MyRandomScreen())
        @test count(j -> md.sector[j] == "S9", pick(scr, md, 1000; X=X, γ_t=γ, N=3, seed=3)) == 2
    end
    g = [isodd(j) ? -0.01j : 0.02 for j in 1:48]; g[13:24] .= 0.05      # sector 2: no γ < 0
    s = short_candidates(g, md0; N=3, exclude=[11])
    @test filter(j -> j <= 12, s) == [5, 7, 9] && !any(13 .<= s .<= 24) && issorted(s)
    @test length(short_candidates(g, md0; N=10)) == 6 + 6 + 6            # all γ < 0 names of sectors 1, 3, 4
end
```

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_screens.jl`. Expected: FAIL.
- [ ] **Step 3: Implement** `Screens.jl`, add the include, exports and the `runtests.jl` entry.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: bandit, γ and random ticker screens; 130/30 short candidates"`

---

### Task 8: Study settings, config file, and the study context

**Files:**
- Create: `code/src/StudySettings.jl`, `code/src/Context.jl`, `config/study.toml`
- Test: `code/test/test_study_config.jl`

**Interfaces:**
- Consumes: Tasks 2, 3, 5, 6, 7.
- Produces:
  - `MyStudySettings`, an inner zero-argument constructor with the defaults below. Field names are ASCII so TOML keys equal field names **(plan decision)**:

    | Fields | Defaults |
    |---|---|
    | `start, stop, train_start, train_stop::Date` | 2025-01-02, 2026-09-04, 2014-01-03, 2024-12-31 |
    | `W0` | 100_000.0 |
    | `xi_short, xi_long, xi_quantile, gtilde_window, beta_floor` | 21, 63, 0.99, 252, 0.1 |
    | `ewls_half_life, ewls_prior_weight` | 63.0, 63.0 |
    | `omega_window, lambda` | 126, NaN (= calibrated) |
    | `T, z, D_max, cash_revisit, n_paths` | 21, 1.96, 0.08, 21, 1000 |
    | `N_per_sector, repick_interval, repick_on_band_exit` | 3, 21, true |
    | `bandit_lookback, bandit_horizon, bandit_eps_floor, bandit_iters_per_arm, bandit_min_iters, bandit_max_iters` | 252, 21, 0.05, 50, 500, 5000 |
    | `w_max, vol_cap_annual, s_max, turnover_frac` | 0.10, NaN (= SPY 2014–2024), 1.0, 0.10 |
    | `ces_eta, sg_baseline, ls_long, ls_short, ls_net, borrow_rate` | 0.5, 0.01, 1.30, 0.30, 1.00, 0.0075 |
    | `half_spread_bps, impact_kappa` | 5.0, 0.001 |
    | `hedge, beta_ref, kappa_beta, beta_min, pool_H, downday_lookback` | true, 1.0, 1.0, 0.0, 5, 252 |

  - `build(MyStudySettings, data)`: overrides the given fields; unknown fields throw `ArgumentError`. It also validates (`ArgumentError`):
    - windows and counts ≥ 1;
    - `xi_short < xi_long`;
    - `0 < xi_quantile < 1`;
    - `beta_min ≥ 0` (needed so the min-variance down-scaling keeps the cap);
    - `0 < w_max ≤ 1`;
    - `ces_eta > 0`;
    - `train_stop < start ≤ stop`.
  - `with_settings(s::MyStudySettings; kw...)::MyStudySettings`: a copy with fields replaced.
  - `load_study(path::AbstractString)::NamedTuple{(:settings, :grid)}`:
    - Every table except `[grid]` is flattened into settings keys; a key repeated across tables or unknown throws.
    - Each value is converted to its field's type (e.g. `63` → `63.0` for `ewls_half_life`).
    - `"calibrated"` for `lambda` or `vol_cap_annual` → `NaN`.
    - `grid` = `(n_seeds::Int, base_seed::Int, allocators::Vector{String}, screens::Vector{Symbol}, hedge::Vector{Bool})`.
  - `context_key(s::MyStudySettings)::NamedTuple`, the settings the context depends on: `start, stop, train_start, train_stop, ewls_half_life, ewls_prior_weight, xi_short, xi_long, xi_quantile, gtilde_window, beta_floor, pool_H, downday_lookback, bandit_horizon`.
  - `MyStudyContext` with fields:
    - `md`, `calib::MyCalibration`, `settings::MyStudySettings`
    - `rows::UnitRange{Int}`, `G::Float64`
    - `ξ::Vector{Float64}`, `g̃::Vector{Float64}` (per md row)
    - `sim::MyOnlineSIM`
    - `γ::Matrix{Float64}` (run rows × tickers)
    - `pool::Vector{Vector{Int}}` (per run row)
    - `X::Matrix{Float64}`
  - `build_context(md, calib, s)::MyStudyContext`, computed in this order:
    1. G = `xi_gain(spy closes over the training rows; xi_short, xi_long, xi_quantile)`.
    2. ξ = `xi_series(md.spy_close; G)`; g̃ = `gtilde_series(md.spy_close; window=gtilde_window)`.
    3. `sim = online_sim(md, calib; start, stop, ewls_half_life, ewls_prior_weight)`; `rows = sim.rows`.
    4. γ[s, :] = `gamma_weights(sim.α[s,:], sim.β[s,:], g̃[t], ξ[t]; beta_floor)` with t = rows[s].
    5. pool[s] = `hedge_pool(sim.β[s,:], down_day_beta(md, t; lookback=downday_lookback); H=pool_H)`.
    6. X = `excess_growth_windows(md; horizon=bandit_horizon)`.
    7. Throws `ArgumentError` when `first(rows) ≤ max(omega_window, downday_lookback)`.
  - `config/study.toml`: every default above, in tables `[window] [market] [sim] [covariance] [triggers] [screens] [limits] [allocators] [costs] [hedge] [grid]`, with `lambda = "calibrated"`, `vol_cap_annual = "calibrated"`, TOML dates, and the grid `n_seeds = 30`, `base_seed = 5660`, `allocators = ["CD", "CES", "SG", "MinVar", "Tangent", "LS"]`, `screens = ["bandit", "gamma", "random"]`, `hedge = [true, false]`. Each line carries a short comment with units.

- [ ] **Step 1: Write the failing tests**

```julia
using Test, Dates, VLINFORMS2026
include("helpers.jl")
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
same(a, b) = all(isequal(getfield(a, f), getfield(b, f)) for f in fieldnames(MyStudySettings))
@testset "settings and config file" begin
    s = MyStudySettings()
    @test (s.xi_short, s.xi_long, s.gtilde_window, s.N_per_sector, s.w_max, s.ces_eta, s.pool_H) == (21, 63, 252, 3, 0.10, 0.5, 5)
    @test isnan(s.lambda) && isnan(s.vol_cap_annual) && s.hedge && s.repick_on_band_exit
    cfg = load_study(joinpath(REPO, "config", "study.toml"))
    @test same(cfg.settings, s)                                        # the file holds the spec defaults
    @test cfg.grid.n_seeds == 30 && cfg.grid.base_seed == 5660 && length(cfg.grid.allocators) == 6
    @test cfg.grid.screens == [:bandit, :gamma, :random] && cfg.grid.hedge == [true, false]
    bad = joinpath(mktempdir(), "bad.toml"); write(bad, "[market]\nxi_shrot = 21\n")
    @test_throws ArgumentError load_study(bad)
    @test_throws ArgumentError build(MyStudySettings, (beta_min=-0.1,))
    @test_throws ArgumentError build(MyStudySettings, (N_per_sector=0,))
    @test with_settings(s; hedge=false).hedge == false && s.hedge
end
@testset "study context" begin
    md = synthetic_md(ρ=0.3)
    fit = fit_base_sim(md, md.dates[1], md.dates[800])
    c = build(MyCalibration, (base_sim=fit.params, σ_m=fit.σ_m, m_m=fit.m_m, v_m=fit.v_m, λ=0.5,
        train_start=md.dates[1], train_stop=md.dates[800]))
    s = build(MyStudySettings, (train_start=md.dates[1], train_stop=md.dates[800], start=md.dates[801], stop=md.dates[900]))
    ctx = build_context(md, c, s)
    @test ctx.rows == 801:900 && ctx.G == xi_gain(md.spy_close[1:800]; short=21, long=63, q=0.99)
    @test ctx.γ[10, :] == gamma_weights(ctx.sim.α[10, :], ctx.sim.β[10, :], ctx.g̃[810], ctx.ξ[810]; β_floor=0.1)
    @test ctx.pool[10] == hedge_pool(ctx.sim.β[10, :], down_day_beta(md, 810; lookback=252); H=5)
    q = build_context(perturb_after(md, md.dates[850]), c, s)          # canary
    @test q.ξ[1:850] == ctx.ξ[1:850] && q.g̃[1:850] == ctx.g̃[1:850]
    @test q.γ[1:50, :] == ctx.γ[1:50, :] && q.pool[1:50] == ctx.pool[1:50]
    @test_throws ArgumentError build_context(md, c, with_settings(s; start=md.dates[200], train_stop=md.dates[150]))
end
```

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_study_config.jl`. Expected: FAIL.
- [ ] **Step 3: Implement** both files and write `config/study.toml`; add the includes, exports and the `runtests.jl` entry.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: study settings, config/study.toml and the study context"`

---

### Task 9: Growth-based scorecard

**Files:**
- Create: `code/src/GrowthScorecard.jl`
- Test: `code/test/test_growth_scorecard.jl`

**Interfaces:**
- Produces: `growth_scorecard(equity::Vector{Float64}, spy_close::Vector{Float64}, g_f::Vector{Float64}; Δt=1/252)::NamedTuple{(:sharpe, :max_drawdown, :alpha, :alpha_t, :beta, :realized_growth, :terminal, :vol)}` (§12):
  - g_W,k = log(W_k/W_{k−1})/Δt, using `g_f[k]` for the day ending at k.
  - Sharpe = mean(g_W − g_f)/(√Δt·std(g_W − g_f)).
  - Alpha: OLS of (g_W − g_f) on (g_SPY − g_f) with intercept. `alpha` (yr⁻¹) is the intercept, `beta` the slope; `alpha_t` uses the classical SE, as in the old `scorecard`. `NaN` conventions as in the old `scorecard`.
  - realized_growth = log(W_end/W_1)/((N−1)Δt).
  - terminal = W_end (USD); vol = std(g_W)·√Δt.
  - Input checks as in the old `scorecard`.

- [ ] **Step 1: Write the failing test**

```julia
using Test, Statistics, VLINFORMS2026
@testset "growth scorecard" begin
    n = 300; Δt = 1/252
    spy = 100 .* exp.(cumsum(vcat(0.0, 0.01 .* sin.(0.3 .* (1:n-1)))))
    W = 1e5 .* (spy ./ spy[1]) .^ 2 .* exp.(0.05 .* (0:n-1) .* Δt)       # g_W = 2·g_SPY + 0.05
    sc = growth_scorecard(W, spy, fill(0.04, n))
    gW = diff(log.(W)) ./ Δt
    @test sc.sharpe ≈ mean(gW .- 0.04) / (std(gW .- 0.04) * sqrt(Δt))
    @test sc.sharpe ≈ mean(diff(log.(W)) .- 0.04Δt) / std(diff(log.(W)) .- 0.04Δt) * sqrt(1 / Δt)   # realized_scorecard
    @test sc.beta ≈ 2.0 rtol=1e-10 && sc.alpha ≈ 0.09 rtol=1e-8                                    # 0.05 + g_f
    @test sc.realized_growth ≈ log(W[end] / W[1]) / ((n - 1) * Δt) && sc.terminal == W[end]
    @test sc.vol ≈ std(gW) * sqrt(Δt)
    @test growth_scorecard([100.0, 120.0, 90.0, 130.0, 110.0], fill(100.0, 5) .* [1, 1.01, 1.0, 1.02, 1.01], zeros(5)).max_drawdown ≈ 0.25
    @test !any(occursin("return", String(k)) for k in keys(sc))
end
```

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_growth_scorecard.jl`. Expected: FAIL.
- [ ] **Step 3: Implement**, add the include, export and the `runtests.jl` entry.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: growth-rate scorecard (Sharpe, drawdown, alpha vs SPY)"`

---

### Task 10: The study backtest loop

**Files:**
- Create: `code/src/StudyBacktest.jl`
- Modify:
  - `code/src/Backtest.jl`: give `_bt_fill!`, `_bt_cash_after_buys` and `_bt_execute!` keyword arguments `half_spread_bps=5.0, κ=0.001` and pass them to `trade_cost`. The defaults keep `run_backtest` unchanged.
  - `code/test/helpers.jl` (study fixtures)
- Test: `code/test/test_study_backtest.jl`

**Interfaces:**
- Consumes: Tasks 4–9 and the Backtest.jl helpers.
- Produces:
  - `MyStudyRun` with fields `settings::MyStudySettings`, `allocator::String` (one of `"CD","CES","SG","MinVar","Tangent","LS"`), `screen::Symbol` (`:bandit, :gamma, :random`), `seed::Int` (≥ 0) and `label::String` (default `""`). Its `build` validates.
  - `make_rule(allocator::String, s::MyStudySettings)::AbstractPortfolioRule`, using `ces_eta`, `sg_baseline` and the `ls_*` caps.
  - `make_screen(screen::Symbol, s::MyStudySettings)::AbstractScreen`, with `MyBanditScreen` built from the `bandit_*` settings.
  - `MyStudyResult` with fields:
    - `dates`, `equity`, `cash`, `borrow_fees` (USD).
    - `trades::DataFrame` (columns as `MyBacktestResult.trades`).
    - `decisions::DataFrame` with `date, reason, status, n_names, invested_frac, μ_T, σ_T, beta_target, beta_max, cap_binds, hedge_weight`.
    - `repicks::DataFrame` with `date, trigger (:initial | :timer | :band_exit), basket, pool, shorts (Vector{String} each), changed::Bool`.
    - `daily::DataFrame` with `date, xi, beta_max, portfolio_beta, hedge_weight, pool_size`.
    - `targets::Dict{Date,Dict{String,Float64}}`, `projections::Dict{Date,MyProjection}`.
  - `_study_problem_columns(rule, basket::Vector{Int}, pool::Vector{Int}, shorts::Vector{Int}, γ_row::AbstractVector)::Tuple{Vector{Int},Vector{Bool}}`. Returns sorted unique columns and their hedge flags:
    - long-only rules: basket ∪ pool;
    - `MyLongShortCDRule`: {basket with γ > 0} ∪ pool ∪ {shorts with γ < 0}.
  - `_study_problem(ctx, run, t::Int, cols::Vector{Int}, hedge::Vector{Bool}, held::Vector{Int}, V::Float64)::MyAllocationProblem`. With s = t − first(ctx.rows) + 1 and S = close[t, cols]:
    - γ = ctx.γ[s, cols].
    - α, β = ctx.sim at row s.
    - μ_g = `expected_growth(rule, α, β, ctx.g̃[t])`.
    - E = `residual_matrix(growth_rates(close[t−Ω:t, cols]), growth_rates(spy_close[t−Ω:t]), α, β)`, with Ω = `omega_window`.
    - σ̂ = `std.(eachcol(E))`.
    - Σ_g = `growth_covariance(build(MyShrunkResidualCovariance, (λ=λ, window=Ω)), β, σ̂, ctx.sim.σ_m[s]; residuals=E)`. This equals σ²_M ββᵀ + (1−λ)Ω̂_ε + λ·diag(Ω̂_ε) (§7).
    - λ = `isnan(lambda) ? calib.λ : lambda`.
    - σ_g_max = `isnan(vol_cap_annual) ? calib.σ_m : vol_cap_annual/√Δt`.
    - Constraint set: `w_max`, `s_max`, `K_turnover = turnover_frac·V`, `c̄ = half_spread_bps·1e-4·mean(S)`, `n_prev = held[cols]`, `ε = 1e-3`, `β = β`, `β_max = hedge_on ? beta_cap(ctx.ξ[t]; beta_ref, kappa_beta, beta_min) : Inf`.
    - Problem: prices = S, W = V, g_f = md.g_f[t], `hedge` flags.
  - `run_study(ctx::MyStudyContext, run::MyStudyRun)::MyStudyResult`. Runs on rows `ctx.rows`, with equity starting at `W0` in cash at the start close. Each day t, in order:
    1. **Fill** the queued orders at open[t] with `_bt_fill!(…; half_spread_bps, κ=impact_kappa)`.
    2. **Accrue** positive cash by exp(g_f·Δt); charge `borrow_fee(short; rate=borrow_rate)` (skipped on the first day).
    3. **Mark** V = cash + holdings at close[t].
    4. **Trigger:** τ += 1 (after the first day); `reason = check_trigger(trig, mpc)`, with `mpc` from `T, z, n_paths, D_max, cash_revisit`. `wealth_peak` is never reset, `initial` is set on the first day, and `trig.basket_changed` is never set.
    5. **Re-pick** when it is the first day (`:initial`), when t − last_repick ≥ `repick_interval` (`:timer`), or when `reason == :band_exit && repick_on_band_exit` (`:band_exit`):
       - basket = `pick(screen, md, t; X=ctx.X, γ_t=ctx.γ[s,:], N=N_per_sector, seed=run.seed)`.
       - pool = `hedge ? ctx.pool[s] : Int[]`.
       - shorts = `allocator == "LS" ? short_candidates(ctx.γ[s,:], md; N=N_per_sector, exclude=pool) : Int[]`.
       - `changed` = the set basket ∪ pool ∪ shorts differs from the last one; `last_repick = t`.
       - If `changed && reason == :in_spec`, then `reason = :basket_change`.
       - Push a `repicks` row.
    6. **Decide** if `reason != :in_spec`:
       - cols, hedge flags from `_study_problem_columns(rule, basket, pool, shorts, ctx.γ[s,:])`.
       - `p = _study_problem(...)`; `r = allocate(rule, p)`. Non-finite shares → `_result(p, n_prev, :solver_failed)` (R29).
       - Projection: `project_gbm(r.n, S, V − r.n⋅S, p.μ_g, p.Σ_g, g_f[t]; T, N=n_paths, rng=MersenneTwister(_bt_projection_seed(seed, date)))`.
       - Record the decision row with `beta_target = r.w⋅β`, `beta_max`, `cap_binds = isfinite(beta_max) && beta_target ≥ beta_max − 1e-6`, and `hedge_weight = Σ r.w[hedge]`. Record targets and the projection.
       - Queue orders (held columns outside `cols` target 0) unless t is the last day.
       - Set τ = 0, the projection, `initial = false`, and `last_alloc_was_cash = (status == :no_preferred)`.
    7. **Daily row:** `xi = ξ[t]`; `beta_max` (the cap value when hedge is on, else `Inf`); `portfolio_beta = Σ held·close·β_online/V`; `hedge_weight` = held value in pool names / V; `pool_size`.
- Test fixtures (append to `helpers.jl`):
  - `study_md(; seed=5660)`: synthetic market on 2014–2019 weekdays. 4 sectors × 12 names, with β = range(−0.6, 1.6; length=12) repeated per sector, so names 1, 13, 25 and 37 have negative β. Residual SD 3 yr⁻¹, SPY growth N(0.1, 2.5²) yr⁻¹, volume 1e6, g_f = 0.04, opens at the geometric midpoints of the closes. Built like `synthetic_md`.
  - `study_ctx(; kw...)`:
    - calibration from `fit_base_sim` on 2014-01-01 → 2017-12-29 with `λ = 0.5`;
    - settings `start` = first trading day ≥ 2018-07-02, `stop` = 2018-12-31, train window as above, overridden by `kw`;
    - returns `(md, ctx)`.
  - `study_run(ctx, allocator, screen; seed=5660)`.

- [ ] **Step 1: Write the failing tests**

```julia
using Test, Dates, Statistics, LinearAlgebra, DataFrames, Random, VLINFORMS2026
include("helpers.jl")
@testset "study loop: accounting and first day" begin
    md, ctx = study_ctx()
    res = run_study(ctx, study_run(ctx, "CD", :gamma))
    @test res.dates == md.dates[ctx.rows] && res.equity[1] == 100_000.0
    @test all(isapprox.(res.equity, res.cash .+ held_value(res, md); rtol=1e-10)) && all(res.cash .>= 0)
    @test res.decisions.reason[1] == :initial && res.repicks.trigger[1] == :initial && res.repicks.date[1] == res.dates[1]
    @test minimum(res.trades.date) == res.dates[2]
    @test run_study(ctx, study_run(ctx, "CD", :gamma)).equity == res.equity                     # deterministic
end
@testset "study loop: the first decision is the documented problem" begin
    md, ctx = study_ctx(); run = study_run(ctx, "CD", :gamma); res = run_study(ctx, run)
    t = first(ctx.rows); basket = [findfirst(==(x), md.tickers) for x in res.repicks.basket[1]]
    cols, hedge = VLINFORMS2026._study_problem_columns(MyCobbDouglasRule(), basket, ctx.pool[1], Int[], ctx.γ[1, :])
    p = VLINFORMS2026._study_problem(ctx, run, t, cols, hedge, zeros(Int, length(md.tickers)), 100_000.0)
    @test p.γ == ctx.γ[1, cols] && p.μ_g ≈ ctx.sim.α[1, cols] .+ ctx.sim.β[1, cols] .* ctx.g̃[t]
    @test p.constraints.β == ctx.sim.β[1, cols] && p.constraints.β_max == beta_cap(ctx.ξ[t])
    @test p.constraints.σ_g_max == ctx.calib.σ_m && p.constraints.c̄ ≈ 5e-4 * mean(md.close[t, cols])
    E = residual_matrix(growth_rates(md.close[t-126:t, cols]), growth_rates(md.spy_close[t-126:t]), ctx.sim.α[1, cols], ctx.sim.β[1, cols])
    @test p.Σ_g ≈ ctx.sim.σ_m[1]^2 .* (ctx.sim.β[1, cols] * ctx.sim.β[1, cols]') .+ 0.5 .* cov(E) .+ 0.5 .* Diagonal(var.(eachcol(E)))
    w = allocate(MyCobbDouglasRule(), p).w
    @test all(isapprox(res.targets[res.dates[1]][md.tickers[c]], w[k]; rtol=1e-6) for (k, c) in enumerate(cols))
end
@testset "study loop: problem columns" begin                         # Review Focus 3 and 4
    γ = [0.3, -0.2, 0.1, -0.4, 0.2, -0.1]
    @test VLINFORMS2026._study_problem_columns(MyCobbDouglasRule(), [1, 2, 3], [3, 4], Int[], γ) == ([1, 2, 3, 4], [false, false, true, true])
    @test VLINFORMS2026._study_problem_columns(MyLongShortCDRule(), [1, 2, 3], [4], [5, 6], γ) == ([1, 3, 4, 6], [false, false, true, false])
end
@testset "study loop: re-pick timer and basket changes" begin
    md, ctx = study_ctx(z=1e6, T=10_000, D_max=1.0, repick_on_band_exit=false)
    res = run_study(ctx, study_run(ctx, "CD", :random))
    @test res.repicks.date == res.dates[1:21:end]
    @test Set(res.decisions.reason) ⊆ Set([:initial, :basket_change])
    @test all(d in res.decisions.date for d in res.repicks.date[res.repicks.changed])
end
@testset "study loop: band exits re-pick only when enabled" begin
    md, ctx = study_ctx(z=0.0)
    on = run_study(ctx, study_run(ctx, "CD", :random))
    @test all(d in on.repicks.date for d in on.decisions.date[on.decisions.reason .== :band_exit])
    md, ctx = study_ctx(z=0.0, repick_on_band_exit=false)
    off = run_study(ctx, study_run(ctx, "CD", :random))
    @test any(off.decisions.reason .== :band_exit) && off.repicks.date == off.dates[1:21:end]
end
@testset "study loop: hedge on and off" begin
    for allocator in ("CD", "CES", "SG", "MinVar", "Tangent", "LS")
        md, ctx = study_ctx()
        res = run_study(ctx, study_run(ctx, allocator, :gamma))
        ok = in.(res.decisions.status, Ref((:optimal, :fallback_minvar, :long_only_fallback)))
        @test all(res.decisions.beta_target[ok] .<= res.decisions.beta_max[ok] .+ 1e-6)
        allocator == "LS" || @test all(res.cash .>= 0)
        md, ctx = study_ctx(hedge=false)
        r0 = run_study(ctx, study_run(ctx, allocator, :gamma))
        @test all(isinf, r0.decisions.beta_max) && all(r0.daily.pool_size .== 0)
    end
end
@testset "study loop: 130/30 shorts" begin                           # Review Focus 4
    md, ctx = study_ctx(); res = run_study(ctx, study_run(ctx, "LS", :gamma))
    pos = positions_by_day(res)
    allowed = Set(vcat(res.repicks.shorts...)); pool = Set(vcat(res.repicks.pool...))
    @test all(n >= 0 || t in allowed for p in pos for (t, n) in p)
    @test all(n >= 0 for p in pos for (t, n) in p if t in pool)
end
@testset "study loop: costs come from settings" begin
    md, ctx = study_ctx(half_spread_bps=0.0, impact_kappa=0.0)
    @test sum(run_study(ctx, study_run(ctx, "CD", :gamma)).trades.cost) == 0.0
end
@testset "study loop: look-ahead canary" begin
    md, ctx = study_ctx(); res = run_study(ctx, study_run(ctx, "CD", :bandit))
    d = res.dates[60]
    _, ctxp = study_ctx(; md=perturb_after(md, d))
    rp = run_study(ctxp, study_run(ctxp, "CD", :bandit))
    @test rp.equity[1:60] == res.equity[1:60]
    k = res.decisions.date .<= d
    @test rp.decisions.date[rp.decisions.date .<= d] == res.decisions.date[k]
    @test rp.repicks.basket[rp.repicks.date .<= d] == res.repicks.basket[res.repicks.date .<= d]
end
```

(`study_ctx` accepts `md=` to build the context on a given market and returns that market; every other keyword overrides a setting.)

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_study_backtest.jl`. Expected: FAIL (`run_study` not defined).
- [ ] **Step 3: Implement** `StudyBacktest.jl`, the cost keywords in `Backtest.jl` and the fixtures; add the include, exports and the `runtests.jl` entry.
- [ ] **Step 4: Run** the new file and `test_backtest.jl`. Expected: PASS. If the band-exit testset finds no `:band_exit` at `z = 0.0` because the horizon or drawdown triggers pre-empt it, keep `z = 0.0` and set `T=10_000, D_max=1.0` in both contexts; do not weaken the assertions.
- [ ] **Step 5: Commit** `git commit -m "feat: study backtest loop with in-loop re-picks, hedge pool and β cap"`

---

### Task 11: Study runner, stats and summary

**Files:**
- Create: `code/src/StudyRunner.jl`
- Modify: `code/src/Experiments.jl`. Extract the threaded, resumable, error-catching loop of `run_grid` into `_run_items(todo::Vector{<:Tuple{String,Any}}, run_one::Function; outdir, label_of::Function, key_repr::Function)`, and generalize `_write_error` to take a label and a key repr. `run_grid` keeps its signature, behavior and tests.
- Test: `code/test/test_study_runner.jl`

**Interfaces:**
- Produces:
  - `study_key(run::MyStudyRun)::NamedTuple`: every `MyStudySettings` field plus `allocator, screen, seed`. `label` is excluded.
  - `study_hash(run)::String` = `bytes2hex(sha1(repr(study_key(run))))`.
  - `study_stats(res::MyStudyResult)::NamedTuple`:
    - `turnover` (yr⁻¹, as `run_stats`), `total_cost`, `borrow_fees`;
    - `n_decisions`, `n_band_exit`, `n_drawdown`, `n_horizon`, `n_cash_revisit`, `n_basket_change`;
    - `n_repicks` and `n_basket_changes` (both exclude the initial re-pick);
    - `mean_names`, `mean_invested`, `mean_portfolio_beta` (daily);
    - `cap_bind_share`: share of **decisions** with `cap_binds`; NaN when the cap is off. The spec's "share of days the β cap binds" is read as decision days, the only days the cap is enforced **(plan decision)**;
    - `mean_hedge_weight` (daily), `mean_pool_size` (daily).
  - `run_and_save_study(ctx, run; outdir="results")::String` writes `outdir/runs/<hash>.jld2` with keys `study_key, result, scorecard (growth_scorecard on the run's dates), stats, wall_seconds, label`, through a `.tmp` and a rename.
  - `run_study_grid(md, calib, runs::Vector{MyStudyRun}; outdir="results")::NamedTuple{(:ran, :skipped, :errors)}`:
    - skips existing hashes and dedups;
    - builds one context per `context_key` that still has work;
    - runs through `_run_items`; failures write `<hash>.error.txt`.
  - `tier1_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}`. Iterates allocators × screens × hedge with seeds `base_seed .+ (0:n_seeds-1)`; the γ screen gets `[base_seed]` only. Labels look like `"CD|bandit|hedge=true|seed=5660"`.
  - `fixed_schedule_runs(s, grid)`: the hedge-on subset with `repick_on_band_exit = false`.
  - `rebuild_study_summary(outdir, experiment; hashes=nothing)::DataFrame`:
    - writes `outdir/<experiment>_summary.csv`, one row per run (or per error);
    - columns: `hash, label, status, allocator, screen, seed`, every settings field, `wall_seconds`, the growth-scorecard fields, then the `study_stats` fields;
    - sorted by hash; restricted to `hashes` when given.

- [ ] **Step 1: Write the failing tests**

```julia
using Test, Dates, DataFrames, CSV, VLINFORMS2026
include("helpers.jl")
@testset "study runner" begin
    s = MyStudySettings(); grid = (n_seeds=30, base_seed=5660, allocators=["CD", "CES", "SG", "MinVar", "Tangent", "LS"],
        screens=[:bandit, :gamma, :random], hedge=[true, false])
    runs = tier1_runs(s, grid)
    @test length(runs) == 732 && count(r -> r.screen == :gamma, runs) == 12
    @test length(unique(study_hash.(runs))) == 732
    fx = fixed_schedule_runs(s, grid)
    @test length(fx) == 366 && all(r -> r.settings.hedge && !r.settings.repick_on_band_exit, fx)
    a = runs[1]; b = build(MyStudyRun, (settings=a.settings, allocator=a.allocator, screen=a.screen, seed=a.seed, label="other"))
    @test study_hash(a) == study_hash(b)

    md, ctx = study_ctx()
    r1 = study_run(ctx, "CD", :gamma); r2 = study_run(ctx, "MinVar", :random; seed=3)
    bad = build(MyStudyRun, (settings=with_settings(ctx.settings; bandit_lookback=10_000), allocator="CD", screen=:bandit, seed=1))
    out = mktempdir()
    g1 = @test_logs (:warn, r"run failed") match_mode=:any run_study_grid(md, ctx.calib, [r1, r2, bad]; outdir=out)
    @test length(g1.ran) == 2 && g1.errors == [study_hash(bad)]
    @test isfile(joinpath(out, "runs", study_hash(bad) * ".error.txt"))
    g2 = @test_logs (:warn, r"run failed") match_mode=:any run_study_grid(md, ctx.calib, [r1, r2, bad]; outdir=out)
    @test sort(g2.skipped) == sort(study_hash.([r1, r2]))
    df = rebuild_study_summary(out, "t")
    @test nrow(df) == 3 && isfile(joinpath(out, "t_summary.csv"))
    @test all(c -> c in names(df), ["allocator", "screen", "seed", "hedge", "sharpe", "max_drawdown", "alpha", "n_repicks", "cap_bind_share"])
    @test !any(occursin("return", c) for c in names(df))

    res = run_study(ctx, r1); st = study_stats(res)
    @test st.n_decisions == nrow(res.decisions) && st.n_repicks == nrow(res.repicks) - 1
    @test st.n_basket_changes == count(res.repicks.changed[2:end])
end
```

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_study_runner.jl`. Expected: FAIL.
- [ ] **Step 3: Implement** `StudyRunner.jl` and the `Experiments.jl` refactor; add the include, exports and the `runtests.jl` entry.
- [ ] **Step 4: Run** the new file and `test_experiments.jl`. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: resumable threaded study runner, stats and summary"`

---

### Task 12: Comparison tables and passive references

**Files:**
- Create: `code/src/StudyTables.jl`
- Test: `code/test/test_study_tables.jl`

**Interfaces:**
- Consumes: a summary `DataFrame` (Task 11).
- Produces (each returns a `DataFrame`; `metric` is a column name as a `Symbol`):
  - Every function first normalizes `screen` with `Symbol.(df.screen)`, so a summary read back from CSV (Strings) works like one in memory. Outputs carry `screen` as a `Symbol`.
  - Groups with no γ-screen row (for `gamma_percentile`) or no pairs (for the paired tables) are left out.
  - `group_summary(df; by=[:allocator, :screen, :hedge, :repick_on_band_exit], metrics=[:sharpe, :max_drawdown, :alpha, :alpha_t, :realized_growth])`: per group `n`, and per metric `<m>_median`, `<m>_q25`, `<m>_q75`.
  - `paired_win_rate(df; metric=:sharpe, a=:bandit, b=:random, by=[:allocator, :hedge, :repick_on_band_exit], higher_is_better=true)`: pairs screen `a` with screen `b` by `seed` within each group. Returns `n_pairs, win_rate, median_diff` (a − b).
  - `gamma_percentile(df; metric=:sharpe, by=[:allocator, :hedge, :repick_on_band_exit])`: the γ-screen value and its percentile among the random screen's seeds (ties count half).
  - `hedge_effect(df; metric, by=[:allocator, :screen, :repick_on_band_exit], higher_is_better=true)`: pairs hedge on vs off by seed. Returns `n_pairs, win_rate, median_diff` (on − off).
  - `long_vs_short_hedge(df; metric, higher_is_better=true)`: pairs `allocator == "CD" && hedge` with `allocator == "LS" && !hedge` by `(screen, seed, repick_on_band_exit)` (spec §1 Q4).
  - `passive_references(md, start::Date, stop::Date; W0=100_000.0)`: one `growth_scorecard` row each for SPY buy-and-hold and an equal-dollar buy-and-hold of the whole universe bought at the start close, with a `name` column.

- [ ] **Step 1: Write the failing tests**

```julia
using Test, Dates, DataFrames, VLINFORMS2026
include("helpers.jl")
@testset "comparison tables" begin
    df = DataFrame(allocator=fill("CD", 7), screen=[:bandit, :bandit, :random, :random, :gamma, :bandit, :random],
        hedge=[true, true, true, true, true, false, false], repick_on_band_exit=trues(7), seed=[1, 2, 1, 2, 5660, 1, 1],
        sharpe=[0.9, 0.2, 0.5, 0.4, 0.45, 0.7, 0.6], max_drawdown=[0.1, 0.2, 0.15, 0.12, 0.11, 0.2, 0.25],
        alpha=zeros(7), alpha_t=zeros(7), realized_growth=zeros(7))
    w = paired_win_rate(df; metric=:sharpe)
    row = only(eachrow(w[w.hedge, :]))
    @test row.n_pairs == 2 && row.win_rate == 0.5 && row.median_diff ≈ 0.1      # +0.4, −0.2
    @test only(gamma_percentile(df; metric=:sharpe)[!, :percentile]) == 0.5          # 0.45 between 0.4 and 0.5
    h = hedge_effect(df; metric=:max_drawdown, higher_is_better=false)
    @test only(h[h.screen .== :bandit, :win_rate]) == 1.0                             # 0.1 < 0.2 at seed 1
    g = group_summary(df)
    @test g[(g.screen .== :bandit) .& g.hedge, :sharpe_median] == [0.55]
    md = bt_md(); pr = passive_references(md, md.dates[1], md.dates[end])
    @test pr.name == ["SPY buy-and-hold", "equal-weight universe"]
    @test pr.realized_growth[1] ≈ log(md.spy_close[end] / md.spy_close[1]) / ((length(md.dates) - 1) / 252)
end
```

- [ ] **Step 2: Run** `cd code/test && julia --project=.. test_study_tables.jl`. Expected: FAIL.
- [ ] **Step 3: Implement**; add the include, exports and the `runtests.jl` entry.
- [ ] **Step 4: Run** the file, then the full suite. Expected: all PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: comparison tables and passive references"`

---

### Task 13: Scripts, README and a real-data smoke run

**Files:**
- Create:
  - `scripts/10_calibrate.jl`, `scripts/11_run_study.jl`, `scripts/12_tables.jl`, `scripts/13_figures.jl`
  - `scripts/Project.toml`: the figures environment, with Plots, DataFrames, CSV, JLD2, Dates and `VLINFORMS2026` added through `Pkg.develop(path="code")`.
- Modify: `README.md` (run order, and the plan decisions listed under Global Constraints), `.gitignore` (`results/smoke/`).

**Interfaces:**
- Scripts 10–12 activate `code/` the way `scripts/02_validate_trial.jl` does, and contain only constants and `main()`.
- `10_calibrate.jl`:
  - loads the market (2014-01-03 → 2026-09-04) and `config/study.toml`;
  - runs `calibrate` on the configured training window;
  - writes `results/calibration.jld2` and `results/calibration_summary.toml`: λ, σ_m, vol cap annualized = σ_m·√Δt, m_m, v_m, G for the configured ξ windows, universe size and sector counts;
  - prints the same values.
- `11_run_study.jl [--fixed] [--smoke]`:
  - default: `tier1_runs`, experiment `"tier1"`;
  - `--fixed`: `fixed_schedule_runs`, experiment `"fixed"`;
  - `--smoke`: only CD × γ screen × hedge on/off (2 runs), into `results/smoke/`;
  - calls `run_study_grid` then `rebuild_study_summary(outdir, experiment; hashes=study_hash.(runs))`;
  - prints ran/skipped/errors and the mean wall time per run.
- `12_tables.jl [dir]` (default `results/`):
  - reads `<dir>/tier1_summary.csv` (and `fixed_summary.csv` if present; `--smoke` mode reads `smoke_summary.csv`);
  - writes `results/tables/{group_summary, bandit_vs_random, gamma_percentile, hedge_effect_sharpe, hedge_effect_drawdown, long_vs_short_hedge, passive}.csv`;
  - prints them as Markdown tables.
- `13_figures.jl`: Task 14.

- [ ] **Step 1:** Write `10_calibrate.jl` and run `julia scripts/10_calibrate.jl`. Expected:
  - universe size 373;
  - annualized vol cap ≈ 0.17;
  - λ in {0, 0.1, …, 1};
  - |ξ| ≤ 1 on 99% of training days by construction;
  - the files are written.

  Record the printed values in the commit message.
- [ ] **Step 2:** Write `11_run_study.jl` and run `julia -t auto scripts/11_run_study.jl --smoke`. Expected:
  - 2 runs, 0 errors;
  - `results/smoke/smoke_summary.csv` has 2 rows;
  - hedge-on `cap_bind_share` > 0;
  - hedge-on `mean_pool_size` > 0 (the pool fills from about April 2025).

  Note the wall time per run. If 732 × (wall time) / threads exceeds 12 hours, report it before Task 14.
- [ ] **Step 3:** Write `12_tables.jl` and run `julia scripts/12_tables.jl results/smoke --smoke`. Expected: the CSVs are written under `results/smoke/tables/` without error. Tables that need random seeds come back with zero rows.
- [ ] **Step 4:** Update `README.md`: run order `00 → 10 → 11 → 11 --fixed → 12 → 13`, where outputs land, and the plan decisions. Add `results/smoke/` to `.gitignore`.
- [ ] **Step 5: Commit** `git add scripts README.md .gitignore results/calibration.jld2 results/calibration_summary.toml && git commit -m "feat(scripts): calibration, study runner, tables; smoke run on real data"`

---

### Task 14: Tier 1 compute, tables and figures

**Files:**
- Create: `scripts/13_figures.jl`; outputs `results/tier1_summary.csv`, `results/fixed_summary.csv` (if run), `results/tables/*.csv`, `results/figures/*.png`.

**Interfaces:**
- `13_figures.jl` activates `scripts/` and reads the summary CSVs plus individual run files from `results/runs/`. Before writing any plotting code, load the `dataviz` skill and follow it. Write each figure as a PNG at 1600×900 for slides:
  1. Sharpe by allocator × screen (hedge on): seed distributions for the bandit and random screens, the γ screen as one marker, and SPY buy-and-hold and the equal-weight universe as reference lines.
  2. The same for max drawdown.
  3. Hedge effect: paired on − off differences in Sharpe and max drawdown, by allocator.
  4. One example run (CD, γ screen, hedge on): wealth vs SPY; portfolio β vs β_max(ξ); pool size and hedge weight over time.
  5. ξ and g̃_M over the run.

- [ ] **Step 1:** Run `julia -t auto scripts/11_run_study.jl`. Expected: 732 rows in `results/tier1_summary.csv`, all `status == "ok"`. For any error, read its `.error.txt`, fix the cause under TDD (a failing test first), and re-run; the runner resumes. Do not drop runs.
- [ ] **Step 2:** Check the runs:
  - every run's `realized_growth` is finite;
  - hedge-off runs have `cap_bind_share` NaN and `mean_pool_size` 0;
  - γ-screen runs are identical across repeats (re-run one and compare `equity`).
- [ ] **Step 3:** Run `julia scripts/12_tables.jl`. Expected: every table is written and the bandit-vs-random table has 30 pairs per group.
- [ ] **Step 4:** Write `13_figures.jl`, run it, and inspect each PNG (open it with the Read tool). Fix overlaps or unreadable labels.
- [ ] **Step 5:** If at least 24 hours remain before Fri Oct 30, run `julia -t auto scripts/11_run_study.jl --fixed` (366 runs), then re-run 12 and 13. Otherwise, write in the README that the fixed-schedule run is pending.
- [ ] **Step 6: Commit** `git add results/*_summary.csv results/tables results/figures scripts/13_figures.jl scripts/Project.toml scripts/Manifest.toml && git commit -m "results: Tier 1 study, tables and figures"`

Out of scope for this plan (spec §10, "as time allows, then for the paper"): the one-at-a-time sensitivity reruns. `with_settings` and `run_study_grid` already support them.
