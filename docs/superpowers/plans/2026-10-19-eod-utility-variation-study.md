# End-of-Day Utility-Variation Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Julia package and scripts that replay the constrained Cobb-Douglas / MPC strategy on end-of-day S&P 500 data. The replay covers six portfolio rules × two SIM growth-covariance models, adds a per-sector "beat SPY" bandit basket step, and is validated against the live trial. It produces Sharpe ratio, max drawdown and alpha vs SPY for the Nov 3 INFORMS talk.

**Architecture:**
- One package, `VLINFORMS2026`, in `code/`, using the CHEME-5660 layout: `Include.jl`, `My*` types with `build`.
- One source file per responsibility: data, SIM, covariance, allocators, projection, costs/triggers, metrics, backtest, validation, bandit, experiments.
- Numbered scripts in `scripts/` run the pipeline: calibration → validation gate → Stage 0 tuning → Tier 1 → figures.
- Frozen choices go to `config/frozen/` (committed). Per-run outputs go to `results/runs/` (gitignored), with summary CSVs in `results/` (committed).

**Tech Stack:**
- Julia 1.12.7.
- JuMP + Clarabel for exponential, power and second-order cones, with SCS as fallback, as in the engine.
- DataFrames, CSV, JLD2, Distributions, Statistics, LinearAlgebra, Random, Dates, TOML, SHA, Downloads.
- Plots for figures; Test for tests.

**Spec:** `docs/superpowers/specs/2026-10-19-eod-utility-variation-study-design.md`. Read it first. Section numbers (§) below refer to it.

**Schedule (talk Tue Nov 3; Tier 1 results due Fri Oct 30):**

| Task(s) | Due |
|---|---|
| 1–9 | Sat Oct 24 |
| 10, the validation gate | Sun Oct 25 |
| 11–14 | Tue Oct 27 |
| 15 (compute) | Wed Oct 28 – Thu Oct 29 |
| 16 | Fri Oct 30 |
| 17 | after the talk (paper) |

## Global Constraints

- **Package and layout:**
  - Package `VLINFORMS2026` at `code/` (`code/Project.toml`, `code/src/VLINFORMS2026.jl` includes `code/src/Include.jl`).
  - Scripts activate `code/` and contain only a `main()` plus constants. Every reusable function lives in `code/src`.
- **CHEME-5660 conventions:**
  - Every type is a `mutable struct My…` with an empty `new()` constructor and a `build(::Type{T}, data::NamedTuple)` factory, following the pattern in 5660 `Factory.jl`.
  - A type whose fields have defaults in this plan, such as `MyLongShortCDRule()` or `MyMPCSpec()`, gets an **inner** zero-argument constructor that fills those defaults. Its `build` starts from that constructor and overrides the fields it is given. (Julia 1.12 does not allow an outer zero-argument constructor next to the empty inner `new()`.)
  - Unicode math names.
  - Every function has a docstring that states units.
  - Seeded RNGs, with default seed 5660.
- **Units:**
  - Growth rate g = log(S_t/S_{t−1})/Δt in yr⁻¹, with Δt = 1/252.
  - Σ_g is in yr⁻².
  - The volatility cap is wᵀΣ_g w ≤ σ²_g,max.
  - Annualized vol = σ_g·√Δt. The trial's 0.12 annualized ≡ σ_g,max = 0.12·√252 = 1.9049 yr⁻¹.
- **SIM covariance:**
  - Σ_g = σ_m²ββᵀ + D_g, where D_g = diag(σ_ε²).
  - β and σ_ε are the EWLS estimates at the decision date.
  - σ_m is updated online (user's decision): an exponentially weighted SD of SPY growth with `half_life=63.0` and `prior_weight=63.0`, seeded from the base-fit σ_m, using data through day t only.
  - C1 and C2 share the same σ_m, so they still differ only in residual covariance.
  - The validation replay (Task 10) uses `σ_m_mode = :fixed` to reproduce the trial.
- **C2 covariance:** σ_m²ββᵀ + D_g^{1/2}·R_λ·D_g^{1/2}, where R_λ = (1−λ)·cor(residuals over the trailing 126 trading days) + λ·I.
  - This is the 5660 note's shrinkage, applied on the correlation scale. λ = 1 equals C1 exactly, and the result is always positive semidefinite **(plan decision)**.
- **Data:**
  - Copied by `scripts/00_fetch_data.jl` into gitignored `data/`. Neither `data/` nor `results/runs/` is ever committed.
  - Price files: `SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2`, `SP500-Daily-OHLC-1-2-2025-to-12-31-2025.jld2`, `SP500-Daily-OHLC-1-2-2026-to-09-04-2026.jld2`.
  - File structure: `d["dataset"][ticker]::DataFrame` with columns `open, close, volume, timestamp::DateTime`. Use `Date(timestamp)`.
  - Sectors: `sp500-sectors.csv` (`Symbol`, `GICS Sector`).
  - Risk-free rate: FRED DTB3, converted as g_f = log(1 + DTB3/100) yr⁻¹ and forward-filled. If the file is missing, use a constant 0.04 and print a warning.
- **External sources:** no Alpaca, no JumpHMM, and no network access except the single FRED download.
- **Windows:**
  - calibration: 2014-01-03 → 2024-12-31
  - scored: 2025-01-02 → 2026-09-04
  - Stage 0: warm-up 2014-01-03 → 2015-12-31, tuning 2016-01-04 → 2024-12-31
- **Trial constants (verbatim):**
  - EWLS `half_life=63.0`, `prior_weight=63.0`, applied per update.
  - SPY EMAs with spans 21 and 63, computed on SPY **prices** over the window of closes with dates in (d − 105 calendar days, d], seeded at the window's first close.
  - λ = −20·(ema21/ema63 − 1).
  - g_m = 252·log(SPY_d/SPY_{d−1}).
  - γᵢ = tanh(αᵢ/max(|βᵢ|^λ, 1e-8) + |βᵢ|^(1−λ)·g_m).
  - Floor ε = 1e-3 shares for names with γ ≤ 0.
  - Triggers: z = 1.96, T = 21, D_max = 0.08, cash_revisit = 21, N_paths = 1000.
  - Turnover: c̄·Σ|Δnᵢ| ≤ 0.10·W, with c̄ = 5e-4·mean(prices) (USD/share).
  - Costs per fill: 5 bps half-spread + 0.001·(q/ADV)·price·q, with zero commission. ADV is the trailing 63-day mean volume **(plan decision)**.
- **Timing:** after the close on day t, in this order: update EWLS, then compute γ, μ_g and Σ_g, then check triggers, then solve. Fills happen at the **open of t+1** in whole shares. Cash accrues at g_f.
- **Expected growth:** μ_g,i = αᵢ + βᵢ·ḡ_M, where ḡ_M is the mean SPY growth over the trailing 252 days. `MyTangentRule(zero_alpha=true)` uses βᵢ·ḡ_M.
- **130/30:**
  - Σ long w ≤ 1.30; Σ|short w| ≤ 0.30; net ≤ 1.00 (which is equivalent to cash ≥ 0).
  - Borrow fee 0.0075/yr on short notional, accrued at Δt per day.
  - The sector cap applies to the net sector weight.
- **Stone–Geary baseline:** bᵢ = floor(0.01·W/Sᵢ) shares for every basket name.
- **Bandit:**
  - K = 33; quotas from `assign_quotas` (an engine port); ε floor 0.05; iters = clamp(50·n_arms, 500, 5000).
  - Reward = 1 if the CD-weighted sub-basket's 21-day log return beats SPY's over the same days, else 0.
  - Each arm's value is (1 + wins)/(2 + wins + losses).
  - Retrain on the first trading day of each quarter.
- **Results:**
  - One `results/runs/<hash>.jld2` per run, written atomically (`.tmp`, then `mv`).
  - `results/<experiment>_summary.csv` is rebuilt from `runs/`.

## Review Focus

1. **A ticker with a missing day or a non-positive price** in any file must be dropped from the universe when data loads, so equity is never NaN. Test: Task 1, `universe excludes gappy ticker`.
2. **An infeasible or failed solve** must keep the previous holdings, record the status, and let the run continue. Examples: σ_g,max below the basket's minimum achievable volatility, or Stone–Geary baselines costing more than W. Test: Task 4, `infeasible keeps previous holdings`.
3. **All γ ≤ 0:**
   - The utility rules return ε pins (and the long-short rule returns zeros), with status `:no_preferred`.
   - The backtest sets `last_alloc_was_cash = true`, so `:cash_revisit` fires after 21 days.
   - Tests: Task 4, `all non-positive γ pins at ε`; Task 7, `cash_revisit`.
4. **A drawdown deeper than D_max that persists:** the drawdown trigger fires every day, because `wealth_peak` is never reset (the engine's behavior). The run must complete and record the costs. Test: Task 9, `persistent drawdown completes`.
5. **A sweep killed while writing:** a stray `.tmp` file is ignored, and on resume only missing hashes rerun. Test: Task 12, `partial tmp is rerun`.

---

### Task 1: Package scaffold and market data

**Files:**
- Create: `code/Project.toml`, `code/src/VLINFORMS2026.jl`, `code/src/Include.jl`, `code/src/Types.jl`, `code/src/Factory.jl`, `code/src/Data.jl`, `code/test/runtests.jl`, `code/test/test_data.jl`, `scripts/00_fetch_data.jl`, `.gitignore` (`data/`, `results/runs/`, `*.tmp`), `README.md` (run order)

**Interfaces:**
- Produces:
  - `MyMarketData` with fields:
    - `dates::Vector{Date}`
    - `tickers::Vector{String}` (universe, sorted, SPY excluded)
    - `open`, `close`, `volume::Matrix{Float64}` (days × tickers)
    - `spy_open`, `spy_close::Vector{Float64}`
    - `sector::Vector{String}`
    - `g_f::Vector{Float64}` (yr⁻¹, one per day)
  - `load_market_data(dir::String; start::Date, stop::Date)::MyMarketData`
    - The universe is tickers present in all three files, with a row on every SPY trading day in [start, stop], all opens and closes > 0, and a sector in the CSV.
  - `growth_rates(x::AbstractVecOrMat; Δt=1/252)`: 252·log ratios along rows (length − 1 rows).
  - `index_of(md::MyMarketData, d::Date)::Int`: errors if `d` is not a trading day.
  - `const DATA_FILES::Vector{String}`: the three filenames in Global Constraints.

- [ ] **Step 1: Write `scripts/00_fetch_data.jl` and run it.**
  - Copy the three jld2 files and `sp500-sectors.csv` from `get(ENV, "CHEME5660_DATA", "~/Desktop/julia_work/CHEME-5660-CourseRepository-Fall-2026/code/src/data")` into `data/`.
  - Download `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3` to `data/DTB3.csv`. On failure, warn and continue.
  - Run: `julia scripts/00_fetch_data.jl`. Expected: `ls data/` shows the 3 jld2 files, the sectors CSV and `DTB3.csv`.

- [ ] **Step 2: Write the failing tests in `code/test/test_data.jl`.** `make_fixture(dir)` writes all three `DATA_FILES`, each holding tickers `AAA`, `GAP` and `SPY` with a few weekdays in its year, a sectors CSV with `AAA` and `GAP`, and no DTB3 file. `GAP` is missing 2025-01-06.

```julia
@testset "growth_rates" begin
    @test growth_rates([100.0, 101.0, 99.0]) ≈ 252 .* [log(1.01), log(99/101)]
end
@testset "universe excludes gappy ticker" begin
    md = load_market_data(make_fixture(mktempdir()); start=Date(2025,1,2), stop=Date(2025,1,8))
    @test md.tickers == ["AAA"]
    @test !any(isnan, md.close) && all(md.g_f .== 0.04)
end
@testset "real data" begin
    dir = joinpath(@__DIR__, "..", "..", "data")
    isfile(joinpath(dir, DATA_FILES[3])) || return
    md = load_market_data(dir; start=Date(2014,1,3), stop=Date(2026,9,4))
    @test md.dates[1] == Date(2014,1,3) && md.dates[end] == Date(2026,9,4)
    @test 360 ≤ length(md.tickers) ≤ 430 && "SPY" ∉ md.tickers
    @test Date(2025,7,4) ∉ md.dates && length(md.g_f) == length(md.dates)
end
```

  - The fixture's timestamps use **varying hours** (04:00 in some files or tickers, 05:00 in others), and a test asserts that AAA and SPY rows still align by `Date`.
  - A second fixture variant includes a `DTB3.csv` whose first observation falls after the first trading day and which has a missing (empty) row. The test asserts the leading value, the forward-filled values and g_f = log(1 + DTB3/100).
  - The real-data test also asserts that `open`, `close`, `volume`, `spy_open`, `spy_close` and `g_f` are all finite, and that `dates` is strictly increasing.
- [ ] **Step 3: Run the tests and confirm they fail.** Run: `julia --project=code code/test/test_data.jl`. Expected: FAIL, because `load_market_data` is not defined.
- [ ] **Step 4: Implement `Types.jl` (`MyMarketData`), `Factory.jl` and `Data.jl`, with the dependencies above in `code/Project.toml`.**
- [ ] **Step 5: Run the tests and confirm they pass.** Same command. Expected: all pass.
- [ ] **Step 6: Commit.** `git add code scripts .gitignore README.md && git commit -m "feat: package scaffold and market data loader"`

### Task 2: SIM fit, EWLS, market state and preference weights

**Files:** Create `code/src/SIM.jl`, `code/test/test_sim.jl`

**Interfaces:**
- Consumes: `MyMarketData`, `growth_rates`.
- Produces:
  - `MySIMParameterEstimate` (fields `α`, `β`, `σ_ε`, `r²`).
  - `estimate_sim(g_m::AbstractVector, g_i::AbstractVector)::MySIMParameterEstimate`: OLS, with σ_ε using the N−2 denominator (5660).
  - `fit_base_sim(md::MyMarketData, start::Date, stop::Date)::NamedTuple{(:params, :σ_m)}`: `params::Dict{String,MySIMParameterEstimate}`; `σ_m` is the SD of SPY growth.
  - `MyEWLSState`, `ewls_init(α₀, β₀, σ_ε₀; half_life=63.0, prior_weight=63.0)`, `ewls_update!(s, g_i, g_m)::Tuple{Float64,Float64,Float64}`: port engine `modified_cobb_douglas_example/code/src/SIM.jl:161-208` unchanged.
  - `MyEWVarState`, `ewvar_init(σ₀; half_life=63.0, prior_weight=63.0)::MyEWVarState`, `ewvar_update!(s, g_m)::Float64` (returns the updated σ_m in yr⁻¹): an exponentially weighted variance around the exponentially weighted mean, with the same η = 2^(−1/half_life) as EWLS and the prior seeding the sufficient statistics.
  - `market_state(spy_close::AbstractVector, dates::Vector{Date}, d::Date)::NamedTuple{(:g_m, :λ, :ḡ_M)}`: g_m and λ exactly as in Global Constraints; `ḡ_M` is the mean of the last 252 SPY growth rates up to d.
  - `preference_weights(α::Vector, β::Vector, g_m::Float64, λ::Float64)::Vector{Float64}`.

- [ ] **Step 1: Write the failing tests.**

```julia
gm = randn(MersenneTwister(5660), 2000) .* 3; gi = 0.5 .+ 1.2 .* gm .+ randn(MersenneTwister(1), 2000)
@test estimate_sim(gm, gi).β ≈ 1.2 atol=0.05
s = ewls_init(0.1, 1.1, 2.0); @test (s.α, s.β, s.σ_ε) == (0.1, 1.1, 2.0)
s = ewls_init(0.0, 0.0, 1.0; half_life=1e12, prior_weight=1e-12)
foreach(t -> ewls_update!(s, gi[t], gm[t]), 1:2000)
@test s.β ≈ estimate_sim(gm, gi).β rtol=1e-6
v = ewvar_init(2.7); @test ewvar_update!(v, 0.0) < 2.7          # a calm day lowers σ_m
v = ewvar_init(1.0; half_life=1e12, prior_weight=1e-12); foreach(x -> ewvar_update!(v, x), gm)
@test ewvar_update!(v, gm[end]) ≈ std(vcat(gm, gm[end]); corrected=false) rtol=1e-3
@test preference_weights([0.2], [0.5], 1.0, 0.0) ≈ [tanh(0.2 + 0.5)]
@test preference_weights([0.2], [0.5], 1.0, 1.0) ≈ [tanh(0.2/0.5 + 1.0)]
# market_state is causal: perturbing prices after d changes nothing
dates = collect(Date(2025,1,1):Day(1):Date(2025,6,30)); spy = 100 .* exp.(cumsum(0.001 .+ 0.01 .* randn(MersenneTwister(2), length(dates))))
d = Date(2025,5,1); spy2 = copy(spy); spy2[findfirst(==(d), dates)+1:end] .*= 2
@test market_state(spy, dates, d) == market_state(spy2, dates, d)
```

- **Value tests (added after review).** These use trading-day dates (weekdays only) so that the 105-day window crosses weekends:
  - `estimate_sim` on a hand-computable case asserts σ_ε with the N−2 denominator, and r².
  - `market_state` asserts λ against a hand-computed EMA ratio.
  - It asserts that ḡ_M equals the mean of exactly the last 252 growth rates when more than 252 exist, and of all of them when fewer exist.
  - It asserts that the EMA window starts at the first close with a date after d − 105 days.
- **Performance:** `market_state` slices only the last ≤ 253 closes for ḡ_M. It must not rebuild the full history on each call.
- [ ] **Step 2: Run them and confirm they fail.** `julia --project=code code/test/test_sim.jl` → FAIL (undefined).
- [ ] **Step 3: Implement `SIM.jl`.**
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit.** `git commit -am "feat: SIM fit, EWLS, market state, preference weights"` (also `git add` the new files)

### Task 3: SIM growth-covariance models and shrinkage choice

**Files:** Create `code/src/Covariance.jl`, `code/test/test_covariance.jl`

**Interfaces:**
- Consumes: `estimate_sim`, `growth_rates`, `MyMarketData`.
- Produces:
  - `abstract type AbstractCovarianceModel`; `MySIMCovariance`; `MyShrunkResidualCovariance` with fields `λ::Float64` and `window::Int` (= 126).
  - `growth_covariance(model, β::Vector, σ_ε::Vector, σ_m::Float64; residuals::Union{Nothing,Matrix{Float64}}=nothing)::Matrix{Float64}` (yr⁻²): exactly the C1 and C2 formulas in Global Constraints. For C2, `residuals` is days × names.
  - `residual_matrix(g::Matrix, g_m::Vector, α::Vector, β::Vector)::Matrix`: εᵢₜ = gᵢₜ − αᵢ − βᵢ·g_mₜ.
  - `choose_shrinkage(md, start::Date, stop::Date; grid=0:0.1:1, n_portfolios=200, seed=5660)::Float64`:
    - Run monthly anchors from `start` + 252 trading days through `stop` − 21.
    - At each anchor, fit an OLS SIM on the trailing 252 days and residual correlations on the trailing 126 days.
    - Draw `n_portfolios` random 33-name baskets (3 names per sector where possible) with Dirichlet(1) weights.
    - Predicted **residual** variance p = wᵀD^{1/2}R_λD^{1/2}w, with D = diag(σ_ε²).
    - Realized r = sample variance of wᵀε_t over the next 21 days. The residuals use the anchor's OLS α and β, so the market-variance forecast stays out of the choice of λ.
    - Loss = mean QLIKE, r/p − log(r/p) − 1. Portfolios with r that is non-positive or not finite are skipped.
  - `shrinkage_loss(md, start, stop; grid=0:0.1:1, n_portfolios=200, seed=5660)::Vector{Float64}` returns the mean loss for each λ.
  - `choose_shrinkage(...)::Float64` returns the argmin of `shrinkage_loss`, with ties going to the larger λ.

- [ ] **Step 1: Write the failing tests.**

```julia
β = [1.0, 1.2, 0.8]; σε = [2.0, 3.0, 2.5]; σm = 2.7
E = randn(MersenneTwister(5660), 500, 3) * cholesky([1 .6 0; .6 1 0; 0 0 1]).U
Σ1 = growth_covariance(MySIMCovariance(), β, σε, σm)
@test Σ1 ≈ σm^2 .* β * β' .+ Diagonal(σε .^ 2)
c2(λ) = growth_covariance(build(MyShrunkResidualCovariance, (λ=λ, window=126)), β, σε, σm; residuals=E)
@test c2(1.0) ≈ Σ1
@test isposdef(c2(0.0)) && c2(0.0)[1,2] - Σ1[1,2] ≈ cor(E)[1,2] * σε[1] * σε[2]
@test sqrt((1/252) * ([.5,.5,0]' * Σ1 * [.5,.5,0])) < 0.5   # annualized vol is sensible
# choose_shrinkage recovers structure on synthetic markets
L8 = shrinkage_loss(synthetic_md(ρ=0.8), Date(2015,1,2), Date(2018,12,31))
@test (0:0.1:1)[argmin(L8)] ≤ 0.3 && (L8[end] - L8[1]) / L8[end] > 0.2     # strong correlation: shrinking hurts
L0 = shrinkage_loss(synthetic_md(ρ=0.0), Date(2015,1,2), Date(2018,12,31))
@test (L0[end] - minimum(L0)) / L0[end] < 0.02                            # no correlation: un-shrinking gains nothing material
# look-ahead canary: prices after `stop` cannot change the loss
@test shrinkage_loss(perturb_after(synthetic_md(ρ=0.8), Date(2018,12,31)), Date(2015,1,2), Date(2018,12,31)) == L8
```

`synthetic_md(; ρ, seed=5660)` is a test helper in `code/test/helpers.jl` (later tasks reuse it), along with `perturb_after(md, d)`, which doubles every price after date `d`. It builds a `MyMarketData` for 2014–2019 (the extra year exists so the look-ahead canary has data to perturb) with 4 sectors of 12 names, β = 1, residual correlation ρ within each sector, and SPY as the market factor.

- [ ] **Step 2: Run the tests and confirm they fail.** `julia --project=code code/test/test_covariance.jl` → FAIL.
- [ ] **Step 3: Implement `Covariance.jl`.**
- [ ] **Step 4: Run the tests and confirm they pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat: SIM growth covariance (C1, C2) and shrinkage choice"`

### Task 4: Shared constraint set and the utility rules (CD, CES, Stone–Geary)

**Files:** Create `code/src/Allocators.jl`, `code/test/test_allocators_utility.jl`, `code/test/helpers.jl` (the `P` and `P5` builders, shared with Task 5), `scripts/dev/make_engine_fixture.jl`, `code/test/fixtures/engine_cd_case.jld2`

**Interfaces:**
- Consumes: `MySIMParameterEstimate`, Σ_g from Task 3.
- Produces:
  - `abstract type AbstractPortfolioRule`; `MyCobbDouglasRule`; `MyCESRule` (`η::Float64`); `MyStoneGearyRule` (`baseline_frac::Float64`).
  - `MyConstraintSet` with fields:
    - `w_max`, `s_max` (1.0 means no sector cap)
    - `σ_g_max` (yr⁻¹), `K_turnover` (USD), `c̄` (USD/share)
    - `sector::Vector{String}`, `n_prev::Vector{Float64}`, `ε` (= 1e-3)
  - `MyAllocationProblem` with fields `γ`, `μ_g` (yr⁻¹), `Σ_g` (yr⁻²), `prices` (USD), `W` (USD), `g_f` (yr⁻¹), `constraints::MyConstraintSet`.
  - `MyAllocationResult` with fields `n::Vector{Float64}` (continuous shares), `w`, `status::Symbol`, `s_star::Float64`.
  - `allocate(rule::AbstractPortfolioRule, p::MyAllocationProblem)::MyAllocationResult`
  - `add_shared_constraints!(m::JuMP.Model, n, p::MyAllocationProblem; signed::Bool=false)`: adds the budget, per-name cap, sector cap, SOC ‖Lᵀw‖ ≤ σ_g_max with LLᵀ = Σ_g + 1e-8I, and the turnover budget. These follow engine `Allocator.jl:115-163` semantics, including its guard that skips the turnover slack variables when that constraint cannot bind.
- **Objectives:**
  - CD: Σ_{γ>0} γᵢ·log nᵢ, using the exponential cone.
  - CES: Σ_{γ>0} γᵢ·tᵢ, with tᵢ ≤ nᵢ^ρ and ρ = 1 − 1/η, using the power cone. It is only needed for η > 1 in Tier 1.
  - Stone–Geary: Σ_{γ>0} γᵢ·log(nᵢ − bᵢ), with nᵢ ≥ bᵢ for every name.
  - Names with γ ≤ 0 are fixed at ε (CD, CES) or at bᵢ (Stone–Geary).
- **Statuses:**
  - `:optimal`, `:no_preferred`, `:infeasible`, `:solver_failed`.
  - On `:infeasible` or `:solver_failed`, return `n = constraints.n_prev`.

- [ ] **Step 1: Generate the engine fixture.** `scripts/dev/make_engine_fixture.jl` runs in the engine environment (`julia --project=$HOME/Desktop/julia_work/modified_cobb_douglas_example/code scripts/dev/make_engine_fixture.jl`):
  - Builds one 8-asset `MyConstrainedCDProblem` with a binding σ_max = 0.12, `w_max` = 0.15, an Σ_annual from `build_sim_covariance`, and nonzero `n_prev`.
  - Saves its inputs and the `solve_constrained_cd` output `n` to `code/test/fixtures/engine_cd_case.jld2`.
- [ ] **Step 2: Write the failing tests.** `P(; kw...)` is a test helper in `code/test/helpers.jl`, shared with Task 5.
  - By default it builds a 3-asset problem: γ = [0.2, 0.5, 0.3], prices = [10, 20, 40], W = 1000, μ_g = 0.1, g_f = 0.
  - Every limit is slack unless overridden: w_max = 1, s_max = 1, σ_g_max = 1e6, c̄ = 0, n_prev = 0.
  - Per-asset defaults (sector, n_prev, μ_g, Σ_g = 4I) are sized from `length(γ)`, so overriding γ changes the asset count.
  - `P5(; kw...)` = `P(; γ=…, prices=[10.,20.,30.,40.,50.], kw...)` for 5 assets.

```julia
γ = [0.2,0.5,0.3]; S = [10.,20.,40.]
@test allocate(MyCobbDouglasRule(), P()).n .* S ≈ 1000 .* γ ./ sum(γ) rtol=1e-5
@test allocate(build(MyCESRule,(η=2.0,)), P()).n ≈ 1000 .* (γ./S).^2 ./ sum(S .* (γ./S).^2) rtol=1e-5
b = floor.(0.01*1000 ./ S); R = 1000 - sum(b .* S)
@test allocate(build(MyStoneGearyRule,(baseline_frac=0.01,)), P()).n ≈ b .+ (γ ./ sum(γ)) .* R ./ S rtol=1e-5
@test maximum(allocate(MyCobbDouglasRule(), P(w_max=0.4)).w) ≤ 0.4 + 1e-6
r = allocate(MyCobbDouglasRule(), P(Σ_g=Diagonal([9.,16.,25.]), σ_g_max=1.5)); @test sqrt(r.w' * Diagonal([9.,16.,25.]) * r.w) ≤ 1.5 + 1e-6
@test sum(allocate(MyCobbDouglasRule(), P(sector=["A","A","B"], s_max=0.5)).w[1:2]) ≤ 0.5 + 1e-6
# all non-positive γ pins at ε
r = allocate(MyCobbDouglasRule(), P(γ=[-0.1,-0.2,-0.3])); @test r.status == :no_preferred && all(r.n .== 1e-3)
# infeasible keeps previous holdings
r = allocate(build(MyStoneGearyRule,(baseline_frac=0.6,)), P(n_prev=[1.,2.,3.])); @test r.status == :infeasible && r.n == [1.,2.,3.]
# reproduces the engine after unit conversion (Σ_g = Σ_annual/Δt, σ_g_max = 0.12/√Δt)
fx = JLD2.load(joinpath(@__DIR__, "fixtures", "engine_cd_case.jld2"))
@test allocate(MyCobbDouglasRule(), problem_from_engine(fx)).n ≈ fx["n"] rtol=1e-4
```

- [ ] **Step 3: Run them and confirm they fail.** `julia --project=code code/test/test_allocators_utility.jl` → FAIL.
- [ ] **Step 4: Implement `Allocators.jl`** (and `problem_from_engine` as a test helper).
- [ ] **Step 5: Run them and confirm they pass.**
- [ ] **Step 6: Commit.** `git commit -m "feat: shared constraints with CD, CES, Stone-Geary rules"`

### Task 5: Min-variance, tangent and 130/30 long-short CD

**Files:** Create `code/src/AllocatorsMeanVariance.jl`, `code/src/AllocatorsLongShort.jl`, `code/test/test_allocators_mv_ls.jl`

**Interfaces:**
- Consumes: `MyAllocationProblem`, `add_shared_constraints!`, `allocate` (Task 4).
- Produces:
  - `MyMinVarRule`; `MyTangentRule` (`zero_alpha::Bool`); `MyLongShortCDRule` (`long_cap=1.30`, `short_cap=0.30`, `net_cap=1.00`).
  - `max_investable(p::MyAllocationProblem; vol_cap::Bool=false)::Float64`: s_dir = min(1, max Σw subject to the per-name cap, sector cap and turnover). With `vol_cap=true` the volatility cone is also included.
  - **Two-fund rule (R24)** for min-variance and tangent:
    1. Choose the mix at Σw = s_dir, without the volatility cone.
    2. Scale it by c = min(1, σ_g,max/√(mixᵀΣ_g mix)) and hold the rest as cash.
    3. If the scaled portfolio violates turnover (not expected at trial constants), keep n_prev with status `:infeasible`.
    - `s_star` = Σw of the returned portfolio.
  - `expected_growth(rule::AbstractPortfolioRule, α::Vector, β::Vector, ḡ_M::Float64)::Vector{Float64}`: α + β·ḡ_M, or β·ḡ_M when `zero_alpha`.
- **Min-variance:** the mix minimizes wᵀΣ_g w with Σw = s_dir; then scale per R24.
- **Tangent:**
  - The mix maximizes (μ_g − g_f)ᵀw / √(wᵀΣ_g w) with Σw = s_dir, via homogenization (y = κw, Σy = s_dir·κ; the caps, sector and turnover rows are homogenized, without the volatility cone). Then scale per R24.
  - A homogenized solve that fails after positive excess was found returns `:solver_failed` with n_prev, not `:fallback_minvar`.
  - If no feasible point has a positive excess return, return the min-variance solution with status `:fallback_minvar`.
- **Long-short CD:**
  - Long side L = {γ > 0}, short side S = {γ < 0}, names with γ = 0 held at 0.
  - Maximize Σ_L γ·log n⁺ + Σ_S |γ|·log n⁻, under the caps in Global Constraints and `add_shared_constraints!(…; signed=true)`.
  - If S is empty, return long-only CD with status `:long_only_fallback`.

- [ ] **Step 1: Write the failing tests.** `P` is as in Task 4; `Σ2 = [4.0 1.0; 1.0 9.0]`.

```julia
@test max_investable(P(w_max=0.15, γ=[0.5,0.5], prices=[10.,20.], Σ_g=Σ2)) ≈ 0.30 atol=1e-6
# R24: with the vol cap binding, tangent keeps its mix and scales down; it no longer collapses to min-variance
rt = allocate(MyTangentRule(), P(Σ_g=Σ2, μ_g=[0.12,0.20], g_f=0.04, γ=[.5,.5], prices=[10.,20.], σ_g_max=1.5))
@test rt.w ./ sum(rt.w) ≈ (Σ2\e) ./ sum(Σ2\e) atol=1e-5
@test sqrt(rt.w' * Σ2 * rt.w) ≈ 1.5 atol=1e-6
rm = allocate(MyMinVarRule(), P(Σ_g=Σ2, γ=[.5,.5], prices=[10.,20.], σ_g_max=1.5))
@test rm.w ./ sum(rm.w) ≈ (Σ2 \ ones(2)) ./ sum(Σ2 \ ones(2)) atol=1e-5
@test maximum(abs.(rt.w .- rm.w)) > 0.05
w = allocate(MyMinVarRule(), P(Σ_g=Σ2, γ=[.5,.5], prices=[10.,20.])).w
@test w ≈ (Σ2 \ ones(2)) ./ sum(Σ2 \ ones(2)) atol=1e-5
e = [0.12, 0.20] .- 0.04
@test allocate(build(MyTangentRule,(zero_alpha=false,)), P(Σ_g=Σ2, μ_g=[0.12,0.20], g_f=0.04, γ=[.5,.5], prices=[10.,20.])).w ≈ (Σ2\e) ./ sum(Σ2\e) atol=1e-5
@test allocate(build(MyTangentRule,(zero_alpha=false,)), P(μ_g=fill(0.0,3), g_f=0.04)).status == :fallback_minvar
r = allocate(MyLongShortCDRule(), P5(γ=[0.5,0.3,0.2,-0.4,-0.1], w_max=0.6))
@test sum(max.(r.w,0)) ≤ 1.30+1e-6 && sum(-min.(r.w,0)) ≤ 0.30+1e-6 && sum(r.w) ≤ 1.0+1e-6
@test all(r.w[1:3] .≥ -1e-9) && all(r.w[4:5] .≤ 1e-9)
@test allocate(MyLongShortCDRule(), P()).status == :long_only_fallback
```

- [ ] **Step 2: Run them and confirm they fail.** `julia --project=code code/test/test_allocators_mv_ls.jl` → FAIL.
- [ ] **Step 3: Implement both files.**
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat: min-variance, tangent, 130/30 long-short CD"`

### Task 6: Multi-asset GBM forward projection

**Files:** Create `code/src/Projection.jl`, `code/test/test_projection.jl`

**Interfaces:**
- Produces:
  - `MyProjection` with fields `μ::Vector{Float64}` and `σ::Vector{Float64}` (USD, length T).
  - `project_gbm(n::Vector{Float64}, prices::Vector{Float64}, cash::Float64, μ_g::Vector{Float64}, Σ_g::Matrix{Float64}, g_f::Float64; T::Int=21, N::Int=1000, rng::AbstractRNG)::MyProjection`
    - Each step: log S += Δt·μ_g + √Δt·L·Z, with LLᵀ = Δt·Σ_g + 1e-12·I.
    - Value at step τ = Σ n·S + cash·exp(g_f·τ·Δt). Negative n means a short position.

- [ ] **Step 1: Write the failing tests.**

```julia
Δt = 1/252; Σg = [0.04/Δt;;]    # 20% annual vol
p = project_gbm([10.0], [100.0], 0.0, [0.10], Σg, 0.0; T=21, N=200_000, rng=MersenneTwister(5660))
m21 = 1000 * exp(0.10*21Δt + 0.5*0.04*21Δt); s21 = m21 * sqrt(exp(0.04*21Δt) - 1)
@test isapprox(p.μ[21], m21; rtol=0.005)
@test isapprox(p.σ[21], s21; rtol=0.02)
@test isapprox(project_gbm([-10.0],[100.0],2000.0,[0.10],Σg,0.0; N=200_000, rng=MersenneTwister(1)).μ[21], 2000 - 1000*exp(0.10*21Δt + 0.5*0.04*21Δt); rtol=0.002)
a = project_gbm([1.,2.],[50.,60.],10.,[.1,.2],[9. 2; 2 16],.04; rng=MersenneTwister(7))
b = project_gbm([1.,2.],[50.,60.],10.,[.1,.2],[9. 2; 2 16],.04; rng=MersenneTwister(7))
@test a.μ == b.μ && a.σ == b.σ
```

- **Correlated two-asset check through `project_gbm` (added after review).** Use n = [1, −1] (and [1, 1]) with a non-zero off-diagonal Σ_g. Compare μ[T] and σ[T] against exact lognormal moments, with V = T·Δt²·Σ_g:
  - m_i = S_i·exp(μ_i·TΔt + ½V_ii)
  - E[S_iS_j] = S_iS_j·exp((μ_i + μ_j)·TΔt + ½(V_ii + V_jj + 2V_ij))
  This catches transposition or scaling errors in the multi-asset loop.
- [ ] **Step 2: Run them and confirm they fail.** `julia --project=code code/test/test_projection.jl` → FAIL.
- [ ] **Step 3: Implement `Projection.jl`.**
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat: multi-asset GBM forward projection"`

### Task 7: Trading costs and rebalance triggers

**Files:** Create `code/src/Costs.jl`, `code/src/Triggers.jl`, `code/test/test_costs_triggers.jl`

**Interfaces:**
- Produces:
  - `trade_cost(q::Int, price::Float64, adv::Float64; half_spread_bps=5.0, κ=0.001)::Float64`: per engine `Costs.jl`; 0 when q = 0.
  - `borrow_fee(short_notional::Float64; rate=0.0075, Δt=1/252)::Float64`
  - `MyMPCSpec` with fields `z=1.96`, `T=21`, `N=1000`, `D_max=0.08`, `cash_revisit_interval=21`.
  - `MyTriggerState` with fields `τ::Int`, `V_t`, `wealth_peak`, `last_alloc_was_cash::Bool`, `projection::Union{Nothing,MyProjection}`, `initial::Bool`, `forced::Bool`, `basket_changed::Bool`, `natural::Bool`.
  - `check_trigger(s::MyTriggerState, spec::MyMPCSpec)::Symbol`. The checks, in priority order:
    1. `:initial`
    2. `:forced`
    3. then, only if `natural`: `:cash_revisit`, `:drawdown`, `:horizon_elapsed`, `:band_exit` (band at index min(τ, T))
    4. `:basket_change`
    5. otherwise `:in_spec`
    - τ ≤ 0 returns `:in_spec`, unless the check is `:initial` or `:forced`.

- [ ] **Step 1: Write the failing tests.**
  - `trade_cost(100, 50.0, 1e6) ≈ 5e-4*50*100 + 0.001*(100/1e6)*50*100`, and `trade_cost(0, 50.0, 1e6) == 0`.
  - `borrow_fee(10_000.0) ≈ 0.0075*10_000/252`.
  - One constructed state per reason returns that reason. For example, `cash_revisit`: `last_alloc_was_cash = true` with τ = 21.
  - Drawdown beats horizon when both hold.
  - `natural = false` with τ = 30 gives `:in_spec`.
  - The band uses the τ-th projection entry.
- [ ] **Step 2: Run them and confirm they fail.** `julia --project=code code/test/test_costs_triggers.jl` → FAIL.
- [ ] **Step 3: Implement both files.**
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat: trading costs and MPC triggers"`

### Task 8: Scorecard metrics

**Files:** Create `code/src/Metrics.jl`, `code/test/test_metrics.jl`

**Interfaces:**
- Produces: `scorecard(equity::Vector{Float64}, spy_close::Vector{Float64}, g_f::Vector{Float64})::NamedTuple{(:sharpe, :max_drawdown, :alpha_ann, :alpha_t, :beta, :ann_return, :ann_vol)}`
  - r_t = V_t/V_{t−1} − 1, and r_f,t = exp(g_f,t·Δt) − 1.
  - Sharpe = √252·mean(r − r_f)/std(r − r_f).
  - Max drawdown = max over t of 1 − V_t/max_{s≤t} V_s.
  - Alpha: OLS of (r − r_f) on (r_SPY − r_f); `alpha_ann` = 252·intercept; `alpha_t` uses the OLS standard error.

- [ ] **Step 1: Write the failing tests.**
  - `scorecard([100.,120,90,130], …).max_drawdown ≈ 0.25`.
  - An equity curve identical to SPY gives `alpha_ann ≈ 0` (atol 1e-10) and `beta ≈ 1`.
  - Equity whose daily returns are 2× SPY's with g_f = 0 gives `beta ≈ 2`.
  - For equity [100, 101, 100, 102, 101] with g_f = 0: r = [0.01, −1/101, 0.02, −1/102] and `sharpe ≈ √252·mean(r)/std(r)`.
- [ ] **Step 2: Run them and confirm they fail.** `julia --project=code code/test/test_metrics.jl` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat: Sharpe, max drawdown, Jensen alpha scorecard"`

### Task 9: Backtest engine

**Files:** Create `code/src/Backtest.jl`, `code/test/test_backtest.jl`

**Interfaces:**
- Consumes: everything in Tasks 1–8.
- Produces:
  - `MyRunConfig` with fields:

    | Field | Meaning |
    |---|---|
    | `rule::AbstractPortfolioRule` | portfolio rule |
    | `covariance::AbstractCovarianceModel` | C1 or C2 |
    | `w_max`, `s_max`, `σ_g_max` | limits |
    | `mpc::MyMPCSpec` | trigger settings |
    | `start::Date`, `stop::Date` | window |
    | `W0` (= 100_000.0) | starting equity |
    | `seed::Int` | RNG seed |
    | `basket_schedule::Vector{Pair{Date,Vector{String}}}` | first entry at `start` |
    | `forced_dates::Vector{Date}` | forced decision dates |
    | `natural_triggers::Bool` (= true) | whether natural triggers can fire |
    | `ewls_calendar::Union{Nothing,Vector{Date}}` (= nothing) | validation only |
    | `base_sim::Dict{String,MySIMParameterEstimate}`, `σ_m::Float64` | base SIM fit; σ_m is the initial value |
    | `σ_m_mode::Symbol` (= `:ewls`) | `:ewls` updates σ_m daily with `ewvar_update!`; `:fixed` holds it constant |
    | `K_turnover_frac` (= 0.10), `borrow_rate` (= 0.0075) | turnover budget, borrow fee |
    | `label::String` | run label |

  - `MyBacktestResult` with fields:
    - `dates`, `equity`, `cash`
    - `trades::DataFrame` (`date`, `ticker`, `qty`, `price`, `cost`)
    - `decisions::DataFrame` (`date`, `reason`, `status`, `n_names`, `invested_frac`, `μ_T`, `σ_T`)
    - `targets::Dict{Date,Dict{String,Float64}}`
    - `projections::Dict{Date,MyProjection}`: the full μ_τ and σ_τ vectors from each decision, needed for Task 16 fig4
  - `run_backtest(md::MyMarketData, cfg::MyRunConfig)::MyBacktestResult`
- **The daily loop at day t:**
  1. Fill pending orders at `open[t]`: whole shares, q = trunc(Int, target) − held (R27: round the TARGET toward zero, then take the difference, so |position| ≤ |target| and ε-pinned names fully exit), with costs. Sells and short sales go first, then buys scaled per R25.
  2. Accrue g_f on positive cash, and charge the borrow fee on short notional.
  3. Mark equity at `close[t]`.
  4. Update EWLS for every ticker in `keys(cfg.base_sim) ∩ md.tickers` from day t's growth. Every basket name must be in `base_sim`; otherwise `run_backtest` throws an `ArgumentError`. When `σ_m_mode == :ewls`, also update σ_m from SPY growth. With `ewls_calendar`, apply one update per calendar entry equal to t, using the latest two trading-day closes on or before that date.
  5. Compute `market_state`, γ, μ_g and Σ_g for the current basket.
  6. Run `check_trigger`.
  7. On a fire:
     - `allocate`, then store the targets.
     - Queue orders for t+1.
     - Run `project_gbm` on the targets, with `rng = MersenneTwister(cfg.seed * 1_000_003 + Dates.value(date))` (R28).
     - Reset τ = 0.
     - Set `last_alloc_was_cash = (result.status == :no_preferred)`. Using the status rather than share counts matters because Stone–Geary's `:no_preferred` holds its 1% baselines, not ε. Min-variance and tangent never set it.

  EWLS starts from `ewls_init` using `base_sim`. C2 residuals use the trailing 126 days of basket growth.

- [ ] **Step 1: Write the failing tests** on a synthetic `MyMarketData` with 3 stocks + SPY, 120 days, and deterministic geometric paths.

```julia
res = run_backtest(md, cfg)
@test all(res.equity .> 0) && res.dates[1] == cfg.start
@test all(isapprox.(res.equity, res.cash .+ held_value(res, md); rtol=1e-10))   # accounting identity
d0 = first(res.decisions.date); @test first(res.trades.date) == md.dates[index_of(md, d0) + 1]   # next-open fills
@test res.trades.price[1] == md.open[index_of(md, d0) + 1, findfirst(==(res.trades.ticker[1]), md.tickers)]
k = 60; md2 = perturb_after(md, k)   # prices after day k doubled
@test filter(r -> index_of(md, r.date) ≤ k, run_backtest(md2, cfg).decisions) == filter(r -> index_of(md, r.date) ≤ k, res.decisions)
@test :forced ∈ run_backtest(md, with(cfg; forced_dates=[md.dates[50]], natural_triggers=false)).decisions.reason
# persistent drawdown completes
r = run_backtest(crash_md(), cfg); @test count(==(:drawdown), r.decisions.reason) ≥ 5 && sum(r.trades.cost) > 0
```

  `held_value`, `perturb_after`, `crash_md` (−40% over 5 days, then flat) and `with` are test helpers.
  - **Regression tests (added after review):**
    - Every position is 0 the day after a `:no_preferred` fill.
    - |position| ≤ |target shares| after every fill.
    - A name entering the basket at a quarterly swap gets the weight implied by EWLS state maintained daily for the whole base_sim universe (R6). Use an uncapped w_max so the result is not hidden by the cap.
- [ ] **Step 2: Run them and confirm they fail.** `julia --project=code code/test/test_backtest.jl` → FAIL.
- [ ] **Step 3: Implement `Backtest.jl`.**
- [ ] **Step 4: Run them and confirm they pass**, then run the full suite: `julia --project=code -e 'using Pkg; Pkg.test()'` → all pass.
- [ ] **Step 5: Commit.** `git commit -m "feat: end-of-day backtest engine"`

### Task 10: Validation against the live trial (gate)

**Files:** Create `scripts/validation/export_trial.jl`, `scripts/02_validate_trial.jl`, `code/src/Validation.jl`, `code/test/test_validation.jl`

**Interfaces:**
- Produces:
  - `compare_targets(sim::Dict{String,Float64}, trial::Dict{String,Float64}; tol=0.01)::NamedTuple{(:pass, :only_sim, :only_trial, :max_abs_diff, :worst)}`: the set of names with w > 1e-3 must match, and every |Δw| ≤ tol.
  - `ewls_calendar_from_log(path::String, trading_days::Vector{Date})::Vector{Date}`: one entry per `[EWLS]` line, giving the latest trading day on or before that line's date. Repeats are kept: 05-29 and 08-21 appear twice, and holiday lines map to the previous trading day.
- **Trial paths**, read-only, from `TRIAL = ~/Desktop/julia_work/constrained-CD-with-MPC-paper-trade-trial`:
  - `artifacts/sim_base.jld2` (`tickers`, `alpha`, `beta`, `sigma_eps`, `sigma_market`, `adv`)
  - `artifacts/frozen_basket.jld2` (`tickers`)
  - `data/trial-log.txt`
  - `data/residuals.csv`
  - `data/tickets/submitted-2026-{05-29,06-29,07-28,08-26,09-01}.jld2` (`.pending.tickers`, `.pending.target_weights`)

- [ ] **Step 1: Write the failing tests.**
  - `compare_targets` returns `pass = true` for identical dicts.
  - It fails on a 0.02 difference, and reports `only_sim` for an extra name.
  - `ewls_calendar_from_log` on this fixture:

```julia
log = """
2026-05-28 16:15:25 [EWLS] updated=33  skipped=0
2026-05-29 16:15:30 [EWLS] updated=33  skipped=0
2026-05-29 18:50:20 [EWLS] updated=33  skipped=0
2026-07-02 16:15:27 [EWLS] updated=33  skipped=0
2026-07-03 16:15:27 [EWLS] updated=33  skipped=0
2026-07-06 16:15:27 [EWLS] updated=33  skipped=0
"""
td = Date.(["2026-05-28","2026-05-29","2026-06-01","2026-07-01","2026-07-02","2026-07-06"])
@test ewls_calendar_from_log(write_tmp(log), td) == Date.(["2026-05-28","2026-05-29","2026-05-29","2026-07-02","2026-07-02","2026-07-06"])
```
- [ ] **Step 2: Run them and confirm they fail.** `julia --project=code code/test/test_validation.jl` → FAIL.
- [ ] **Step 3: Implement `Validation.jl` and `export_trial.jl`.** `export_trial.jl` runs with `julia --project=$TRIAL/code scripts/validation/export_trial.jl`, because the ticket types are defined in the trial's `LiveCD` package. It writes:
  - `data/validation/trial_targets.csv` (`date`, `ticker`, `weight`)
  - copies of the artifacts, the log and residuals.csv in `data/validation/`
- [ ] **Step 4: Run the tests and confirm they pass.**
- [ ] **Step 5: Implement and run `scripts/02_validate_trial.jl`.** Settings:
  - CD, C1, σ_m = `sigma_market` with `σ_m_mode = :fixed` (as in the trial), `base_sim` built from `sim_base`
  - σ_g_max = 0.12·√252, w_max = 0.15, s_max = 1
  - basket = the trial's frozen 33. This is the only place they are used. Any name missing from the universe is dropped and listed in `allocation_check.csv`.
  - start 2026-05-20, stop 2026-09-04
  - `forced_dates` = [05-29, 06-29, 07-28, 08-26, 09-01], `natural_triggers=false`, `ewls_calendar` from the log

  It writes `results/validation/allocation_check.csv` (`date`, `pass`, `max_abs_diff`, `worst`, `only_sim`, `only_trial`) and `results/validation/targets_compare.csv` (`date`, `ticker`, `w_sim`, `w_trial`), which Task 16 fig5 uses. It also writes `results/validation/equity_check.csv`: simulated equity next to the trial's `V_t`, with the −$882 options P&L noted.
  - Expected: `pass = true` on all 5 dates.
  - **If any date fails: STOP.** Write a per-name γ comparison table and report it to the user before going further. Likely causes are IEX vs consolidated closes, and the 08-21 10:43 intraday bar. Do not loosen `tol` without the user's sign-off.
- [ ] **Step 6: Run the band check (informational).** Rerun with `natural_triggers=true` and the forced dates removed, once with C1 and once with C2 (λ = 0.5 placeholder until Task 13). Write `results/validation/band_check.csv`, with the trigger dates and reasons next to the trial's.
- [ ] **Step 7: Commit.** `git add code scripts results/validation && git commit -m "feat: validation against live trial"`

### Task 11: Bandit ticker step and random baseline

**Files:** Create `code/src/Bandit.jl`, `code/test/test_bandit.jl`

**Interfaces:**
- Consumes: `estimate_sim`, `market_state`, `preference_weights`, `MyMarketData`.
- Produces:
  - `assign_quotas(sector_groups::Dict{String,Vector{Int}}, K::Int)::Dict{String,Int}`: port of engine `Bandit.jl:6-21`.
  - `beat_spy_reward(arm::Vector{Int}, day::Int, md::MyMarketData, α::Vector, β::Vector, g_m::Vector, λ::Vector; horizon=21)::Int`: 1 if the closed-form-CD-weighted buy-and-hold log return of `arm` over [day, day+21] is greater than log(SPY_{day+21}/SPY_day), else 0.
    - Closed form: nᵢ = B·(γᵢ/Σ_{γ>0}γ)/Sᵢ for γᵢ > 0, and 0 otherwise. If no γ is positive, the arm holds cash, with return 0.
  - `train_sector_bandit(sector_idx::Vector{Int}, q::Int, train_days::UnitRange{Int}, md, α, β, g_m, λ; seed::Int, ε_floor=0.05)::NamedTuple{(:best_arm, :wins, :losses, :n_unique)}`: engine ε schedule; arm value (1+wins)/(2+wins+losses).
  - `quarter_starts(md, start::Date, stop::Date)::Vector{Date}`: the first trading day of each calendar quarter.
  - `bandit_basket_schedule(md::MyMarketData, retrain_dates::Vector{Date}; K=33, seed::Int)::Vector{Pair{Date,Vector{String}}}`
    - At each retrain date R, fit an OLS SIM per ticker on [2014-01-03, R).
    - `train_days` = every d with `md.dates[d+21] < R`.
    - Combine the sector winners into one basket.
  - `random_sector_basket(md::MyMarketData, K::Int; seed::Int)::Vector{String}`: a random draw that meets the `assign_quotas` quotas. `random_matched_schedule` uses it for its first entry, and Task 14 uses it for the Stage 0 baskets.
  - `random_matched_schedule(bandit::Vector{Pair{Date,Vector{String}}}, md; seed::Int)::Vector{Pair{Date,Vector{String}}}`: the first entry is a random draw with the same quotas. Each later entry swaps, per sector, as many names as the bandit swapped, with the new names chosen at random.

- [ ] **Step 1: Write the failing tests.**
  - The quotas sum to 33, and a sector smaller than its quota gets all of its names.
  - On synthetic prices where the arm gains 5% and SPY 1% over 21 days, `beat_spy_reward` returns 1; when the gains are reversed, it returns 0.
  - Causality: doubling every price on or after R leaves `bandit_basket_schedule(…, [R])` unchanged.
  - For each sector and each quarter, the number of names `random_matched_schedule` swaps equals the number the bandit swapped.
- [ ] **Step 2: Run them and confirm they fail.** `julia --project=code code/test/test_bandit.jl` → FAIL.
- [ ] **Step 3: Implement `Bandit.jl`**, with iters = clamp(50·binomial(N_s, q), 500, 5000).
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat: per-sector beat-SPY bandit and matched random baseline"`

### Task 12: Experiment runner

**Files:** Create `code/src/Experiments.jl`, `code/test/test_experiments.jl`

**Interfaces:**
- Consumes: `MyRunConfig`, `run_backtest`, `scorecard`.
- Produces:
  - `config_key(cfg::MyRunConfig)::NamedTuple`: every field except `base_sim`, with `basket_schedule` reduced to its SHA1.
  - `config_hash(cfg)::String` = `bytes2hex(sha1(repr(config_key(cfg))))`.
  - `run_and_save(md, cfg; outdir="results")::String` (returns the hash): writes `outdir/runs/<hash>.jld2` with keys `config_key`, `result`, `scorecard`. It writes to `<hash>.jld2.tmp` and then renames.
  - `run_grid(md, cfgs::Vector{MyRunConfig}; outdir="results")`: `Threads.@threads` over every config whose `<hash>.jld2` is missing.
  - `rebuild_summary(outdir::String, experiment::String)`: writes `outdir/<experiment>_summary.csv` with these columns:
    - `hash, label, rule, covariance, λ, w_max, σ_g_max, s_max, z, T, basket_kind, seed, start, stop`
    - `sharpe, max_drawdown, alpha_ann, alpha_t, beta, ann_return, ann_vol`
    - `turnover, total_cost, borrow_fees`
    - `n_decisions, n_band_exit, n_drawdown, n_horizon, n_cash_revisit, mean_names, mean_invested`

- [ ] **Step 1: Write the failing tests.**
  - The hash is stable across calls, and changes when `w_max` changes.
  - A second `run_grid` with the same configs runs nothing; check that the file modification times are unchanged.
  - **Partial tmp is rerun:** with only a `<hash>.jld2.tmp` present, `run_grid` produces `<hash>.jld2`.
  - The summary has the columns above.
  Use the Task 9 synthetic market data.
- [ ] **Step 2: Run them and confirm they fail.** `julia --project=code code/test/test_experiments.jl` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat: resumable threaded experiment runner"`

### Task 13: Calibration (base SIM fits and λ)

**Files:** Create `scripts/01_calibrate.jl`; outputs `config/frozen/base_sim_2014_2015.csv`, `config/frozen/base_sim_2014_2024.csv` (`ticker, α, β, σ_ε, r²`) and `config/frozen/calibration.toml` (`σ_m_2014_2015`, `σ_m_2014_2024`, `λ_shrink`)

- [ ] **Step 1: Add the base-fit I/O to `code/src/SIM.jl`, test first.**
  - `write_base_sim(path::String, fit::NamedTuple{(:params,:σ_m)})` writes the CSV (`ticker, α, β, σ_ε, r²`).
  - `read_base_sim(path::String)::Dict{String,MySIMParameterEstimate}` reads it back.
  - Test: a round trip in `code/test/test_sim.jl` returns identical values.
  - σ_m is stored in `calibration.toml`.
- [ ] **Step 1b: Write `scripts/01_calibrate.jl`.** It runs `fit_base_sim` for each window, then `choose_shrinkage(md, Date(2014,1,3), Date(2024,12,31))`.
- [ ] **Step 2: Run it.** `julia -t 24 --project=code scripts/01_calibrate.jl`. Expected:
  - both CSVs list every universe ticker
  - 1.5 ≤ σ_m ≤ 4.0 yr⁻¹ (SPY annualized vol 9–25%)
  - λ_shrink ∈ [0, 1], printed
- [ ] **Step 3: Rerun the Task 10 band check with the real λ.**
- [ ] **Step 4: Commit.** `git add config/frozen scripts/01_calibrate.jl results/validation && git commit -m "data: frozen base SIM fits and shrinkage λ"`

### Task 14: Stage 0, tuning σ_g,max and w_max on 2016–2024

**Files:** Create `config/experiments/stage0.toml`, `scripts/03_stage0.jl`; output `config/frozen/stage0.toml`, `results/stage0_summary.csv`

**Grid:**
- σ_g,max ∈ [1.2700, 1.5875, 1.9049, 2.3812, 3.1749] (annualized 0.08–0.20) × w_max ∈ [0.05, 0.10, 0.15, 0.20]
- × all 6 rules (CES η = 2, Stone–Geary 0.01)
- × {C1, C2(λ_shrink)}
- × random-within-sector baskets with seeds 1–5, drawn once at 2016-01-04 with `assign_quotas` and held fixed
- Window 2016-01-04 → 2024-12-31, base SIM 2014–2015.
- Total: 1,200 runs.

- [ ] **Step 1: Time one run.** `julia -t 1 --project=code scripts/03_stage0.jl --time-one` prints the seconds per run. If 1,200 runs × time/24 is more than 8 hours, cut the seeds to 3, record that in `stage0.toml`, and tell the user.
- [ ] **Step 2: Run the grid.** `julia -t 24 --project=code scripts/03_stage0.jl`. Expected: 1,200 rows in `results/stage0_summary.csv`.
- [ ] **Step 3: Select the settings.**
  - For each covariance option, pick the (σ_g,max, w_max) pair with the highest median Sharpe over all (rule, seed) runs.
  - When two pairs are within 0.01 Sharpe, take the one with the lower median max drawdown.
  - Write `config/frozen/stage0.toml`, e.g. `[C1] σ_g_max = …  w_max = …`, and print both pairs.
- [ ] **Step 4: Commit.** `git add config results/stage0_summary.csv scripts/03_stage0.jl && git commit -m "data: Stage 0 frozen σ_g,max and w_max"`

### Task 15: Tier 1 scored runs (2025-01-02 → 2026-09-04)

**Files:** Create `config/experiments/tier1.toml`, `scripts/04_tier1.jl`; outputs `results/tier1_summary.csv`, `results/tier1_table.csv`, `results/benchmarks.csv`

**Runs:**
- 6 rules × {C1, C2} × baskets:
  - bandit seeds 1–10
  - random-matched seeds 1–10
- That is 240 runs, using base SIM 2014–2024 and the `stage0.toml` settings per covariance option.
- Bandit retrain dates: `quarter_starts(md, 2025-01-02, 2026-09-04)`. Each seed's schedule is computed once and cached at `data/cache/bandit_seed<k>.jld2`.

- [ ] **Step 1: Write `scripts/04_tier1.jl`.** It builds the schedules, runs the grid, and runs `rebuild_summary`. It also writes:
  - `tier1_table.csv`: for each (rule, covariance, basket_kind), the median and IQR of Sharpe, max drawdown and alpha_ann, plus the paired win-rate of bandit over random-matched on Sharpe, by seed.
  - `benchmarks.csv`: the `scorecard` of SPY and of an equal-dollar buy-and-hold of the whole universe bought at the 2025-01-02 open.
- [ ] **Step 2: Run it.** `julia -t 24 --project=code scripts/04_tier1.jl`. Expected: 240 rows, no `:solver_failed` count above 1% of decisions (print it), and the benchmark row for SPY shows positive `ann_return` for 2025–26.
- [ ] **Step 3: Commit.** `git add results/tier1_*.csv results/benchmarks.csv config/experiments/tier1.toml scripts/04_tier1.jl && git commit -m "results: Tier 1"`

### Task 16: Talk figures

**Files:** Create `scripts/05_figures.jl`; outputs `talk/figures/*.pdf` and matching `.png`

- [ ] **Step 1: Write the script. It produces:**
  1. `fig1_scorecard.pdf`: a table of median [IQR] Sharpe, max drawdown and alpha by rule × covariance, for the bandit and random-matched baskets.
  2. `fig2_equity.pdf`: for each of the six rules under C2, the median equity path across the 10 bandit seeds, plus SPY.
  3. `fig3_bandit_vs_random.pdf`: paired Sharpe differences by seed.
  4. `fig4_band_calibration.pdf`: realized/projected σ ratio for each segment between decisions, C1 vs C2. This is the 10-19 finding, shown in the replay.
  5. `fig5_validation.pdf`: simulated vs trial target weights on the 5 dates.
- [ ] **Step 2: Run it.** `julia --project=code scripts/05_figures.jl`. Expected: 5 PDFs and 5 PNGs, and the numbers in fig1 equal `results/tier1_table.csv`, which the script asserts.
- [ ] **Step 3: Commit.** `git add talk scripts/05_figures.jl && git commit -m "talk: Tier 1 figures"`

### Task 17 (after the talk, for the paper): Tier 2 sweeps

**Files:** Create `config/experiments/tier2.toml`, `scripts/06_tier2.jl`

- [ ] **Step 1: Sweep one setting at a time on 2016–2024**, around the Stage 0 settings, for both C1 and C2, with 30 random-within-sector seeds:
  - z ∈ {1.96, 2.5, Inf}
  - T ∈ {21, 63}
  - s_max ∈ {1.0, 0.30}
  - CES η ∈ {0.5, 2, 5}. η = 0.5 needs the ρ < 0 conic form; add it with a test against the closed form first.
  - Stone–Geary baseline ∈ {0.005, 0.01, 0.02}
  - tangent `zero_alpha` ∈ {false, true}
- [ ] **Step 2: Freeze the chosen values** in `config/frozen/tier2.toml`, then rescore 2025-01-02 → 2026-09-04 with the Tier 1 baskets.
- [ ] **Step 3: Commit** the summaries.
