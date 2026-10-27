# Synthetic-Market Stress Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a JumpHMM synthetic-market generator for the study's 373 names, run the unchanged study on about
1,000 synthetic histories (recalibrated on each), and produce picker × utility tables for a typical set and two
stress sets in time for the Oct 30 slides.

**Architecture:** Five new source files in `code/src/`. `SyntheticMarket.jl` holds the generator: seeds, the drift
EMA, sector factors, the composition, and the `MyMarketData` builder. `SyntheticCalibration.jl` fits the generator.
`SyntheticChecks.jl` runs checks V1–V9. `SyntheticRunner.jl` picks the paths and runs the study grid on one path,
keeping only summary rows. `SyntheticTables.jl` builds the tables. Scripts 20–24 drive them. The study code is reused
as is; the only edit to existing code is factoring the summary-row builder out of `_study_summary_row`.

**Tech Stack:** Julia 1.12, JumpHMM.jl (varnerlab, pinned), JLD2, DataFrames, CSV, the existing VLINFORMS2026
package.

**Spec:** `docs/superpowers/specs/2026-10-26-synthetic-market-stress-test-design.md` (approved 2026-10-26). Read it
with this plan; section numbers below (§) refer to it.

## Global Constraints

- Growth rates only: g = ln(Sₜ/Sₜ₋₁)/Δt, Δt = 1/252 yr, in yr⁻¹. Never returns.
- No copula and no rank reorder anywhere. Never call `JumpHMM.HybridSingleIndexModel`, `JumpHMM.PortfolioModel`,
  `JumpHMM.SingleIndexModel` or eCornell's `generate_hybrid_scenario`.
- JumpHMM.jl from `https://github.com/varnerlab/JumpHMM.jl.git` at rev `aaed969a4517bd9095ad7daaef82b83a74a03a17`
  (git-tree-sha1 `3b1e4e55527636c213116d65fbd976fa388aae97`). Load it with `import JumpHMM` and qualify every call
  (`JumpHMM.fit`, `JumpHMM.tune`, `JumpHMM.simulate`). Its `fit`, `tune`, `simulate` and `validate` must not enter the
  package namespace unqualified.
- JumpHMM settings for every name and sector fit: `N = 50`, `ν = 5.0`, `rf = 0.0`, `dt = 1/252`,
  `ϵ_range = range(1e-4, 2.5e-2, length=10)`, `λ_range = range(10.0, 120.0, length=10)`, `n_paths = 100`.
- Idiosyncratic floor f = 0.10. R²-preserving branch for real R² ≥ 0.80.
- σ_m², σ_s² and σ_gen,i² are fixed at their real 2014–2024 values (sample variance, N−1), never per path.
- Calendar: the real SPY trading days 2014-01-03 → 2026-09-04. Training window 2014-01-03 → 2024-12-31. Trading
  window 2025-01-02 → 2026-09-04.
- Every random draw comes from a seed derived from the path id (`path_seed`). No unseeded RNG use in the generator.
- Study strategies, `config/study.toml` and the real-data results are untouched. New outputs go only under
  `results/synthetic/`.
- If a §8 check fails: stop and report it to the user. Do not tune to pass.
- Tables only, no figures. Rows are pickers in plain names with name counts, columns are the 7 utilities, and SPY and
  equal-weight rows go at the top. One table per hedge setting. Never pool across pickers or utilities.
- Repo style: `My*` mutable structs with a `build(Type, NamedTuple)` factory; docstrings that state units; exports in
  `code/src/Include.jl`; tests in `code/test/test_*.jl`, each starting with `using Test, Dates, VLINFORMS2026` and
  `include("helpers.jl")`, and each listed in `code/test/runtests.jl`.
- Work on branch `feat/synthetic-stress`, off `feat/ces-eta-sweep`. Commit at the end of each task. Never push or
  merge.
- Schedule (§10): Tasks 1–8 on Tue Oct 27; Tasks 9–10 on Wed Oct 28; Task 11 written Wed and run Thu Oct 29.

## Review Focus

1. **A path whose `calibrate` or `build_context` throws.** That path's rows must be written with `status = "error"`
   and the message, and the run must move on to the next path. Pinned by Task 9's error-row test.
2. **An overnight run killed mid-path.** A half-written path must not count as done on restart. Pinned by Task 9's
   `path_done` test.
3. **A refit name model whose draws are far from the real name's variance** (a leftover split, a degenerate state).
   It must stop script 20 by name, not flow silently into 1,000 paths. Pinned by Task 6's `model_sd_ratio` test and
   gate.
4. **A β path that drifts far negative or far above 1.** The sign must be kept, so the name is a real hedge. Total
   variance must stay at target and prices must stay finite. Pinned by Task 4's β = −3 test and Task 5's
   non-finite-close test.
5. **Different thread counts.** The same path id must give the same market whatever `-t` is. Pinned by Task 5's
   per-name seed test and Task 9's sorted rows.

---

### Task 1: Branch, JumpHMM dependency, stored market model loads

**Files:**
- Modify: `code/Project.toml` (+ `code/Manifest.toml` if tracked), `code/src/Include.jl`, `.gitignore`,
  `code/test/runtests.jl`
- Create: `code/src/SyntheticMarket.jl` (empty module section with a header comment), `code/test/test_synthetic_market.jl`

**Interfaces:**
- Produces: `VLINFORMS2026.JumpHMM` (module, via `import JumpHMM` in `Include.jl` after `using SHA`), and
  `include("SyntheticMarket.jl")` after `include("StudyTables.jl")`.

- [ ] **Step 1: Create the branch and commit the spec and this plan**

```bash
git switch -c feat/synthetic-stress
git add docs/superpowers/specs/2026-10-26-synthetic-market-stress-test-design.md docs/superpowers/plans/2026-10-26-synthetic-market-stress-test.md
git commit -m "docs: synthetic-market stress test design and plan"
```

- [ ] **Step 2: Write the failing test** in `code/test/test_synthetic_market.jl`

```julia
using Test, Dates, VLINFORMS2026, JLD2
include("helpers.jl")
const SURROGATE = expanduser("~/Desktop/julia_work/eCornell-AI-finance-lectures/code/src/data/pretrained-jumphmm-market-surrogate.jld2")
@testset "stored market surrogate loads" begin
    if isfile(SURROGATE)
        m = load(SURROGATE, "model")
        @test m isa VLINFORMS2026.JumpHMM.JumpHiddenMarkovModel
        @test m.rf == 0.0 && m.dt ≈ 1/252 && m.partition.N == 50     # observations are plain growth rates
        r = VLINFORMS2026.JumpHMM.simulate(m, 252; n_paths=1, seed=1)
        @test length(r.paths[1].observations) == 252 && all(isfinite, r.paths[1].observations)
    else
        @test_skip "surrogate file missing"
    end
end
```

Add `include("test_synthetic_market.jl")` to `runtests.jl`.

- [ ] **Step 3: Run it to make sure it fails**

Run: `julia --project=code code/test/test_synthetic_market.jl`
Expected: FAIL. `UndefVarError: JumpHMM not defined in VLINFORMS2026`.

- [ ] **Step 4: Add the dependency**

Run: `julia --project=code -e 'using Pkg; Pkg.add(url="https://github.com/varnerlab/JumpHMM.jl.git", rev="aaed969a4517bd9095ad7daaef82b83a74a03a17")'`

Then check that `code/Manifest.toml` shows `git-tree-sha1 = "3b1e4e55527636c213116d65fbd976fa388aae97"` for JumpHMM.
Add `import JumpHMM` to `Include.jl` and include `SyntheticMarket.jl`. Add `results/synthetic/rows/`,
`results/synthetic/smoke/` and `results/synthetic/*.jld2` to `.gitignore`: the rows run to several hundred MB, and
`generator.jld2` can be rebuilt by script 20.

- [ ] **Step 5: Run the test and the full suite**

Run: `julia --project=code code/test/test_synthetic_market.jl`, then `julia --project=code code/test/runtests.jl`
Expected: PASS. Full suite exit 0; this is V8's baseline, so record the testset count.

- [ ] **Step 6: Commit**

```bash
git add code/Project.toml code/Manifest.toml code/src/Include.jl code/src/SyntheticMarket.jl .gitignore code/test/test_synthetic_market.jl code/test/runtests.jl
git commit -m "build: add JumpHMM.jl pinned to aaed969a for the synthetic market"
```

---

### Task 2: Seeds, growth windows and the drift EMA

**Files:**
- Modify: `code/src/SyntheticMarket.jl`, `code/src/Include.jl` (exports), `code/test/test_synthetic_market.jl`

**Interfaces:**
- Produces:
  - `path_seed(p::Int, stream::Symbol, j::Int=0)::Int`, with streams `:market` → `p`; `:sector` →
    `10^12 + 10^4·p + j`; `:name` → `2·10^12 + 10^4·p + j`; `:drift` → `3·10^12 + p`. It throws `ArgumentError` for
    an unknown stream, for `p > 10^7`, or for `j ≥ 10^4`.
  - `growth_window(dates::Vector{Date}, start::Date, stop::Date)::UnitRange{Int}`: the growth-vector rows between
    consecutive trading days inside [start, stop], i.e. `index(start):(index(stop) − 1)`. Row k is the growth from
    `dates[k]` to `dates[k+1]`. Throws `ArgumentError` if a date is not in `dates`.
  - `ema_rate(h::Real)::Float64 = 1 − 2^(−1/h)` (h in trading days).
  - `drift_path(z::AbstractVector{<:Real}, h::Real, k::Real)::Vector{Float64}`, with
    x₁ = k·√(a/(2−a))·z₁ and xₜ = (1−a)·xₜ₋₁ + a·k·zₜ. The output is linear in k, so calibration can reuse one
    shock bank.

- [ ] **Step 1: Write the failing tests**

```julia
@testset "drift EMA" begin
    z = randn(Xoshiro(1), 400_000); a = ema_rate(252)
    x = drift_path(z, 252, 2.0)
    @test x[1] == 2.0 * sqrt(a / (2 - a)) * z[1]
    @test isapprox(std(x), 2.0 * sqrt(a / (2 - a)); rtol=0.03)           # stationary SD
    @test isapprox(cor(x[1:end-252], x[253:end]), 0.5; atol=0.05)          # (1-a)^252 = 0.5
    @test drift_path(z, 63, 3.0) ≈ 3.0 .* drift_path(z, 63, 1.0)
    @test all(iszero, drift_path(z[1:100], 63, 0.0))
end
@testset "path seeds and growth windows" begin
    ids = vcat(1:200, 1_000_001:1_000_020)
    seeds = [path_seed(p, :market) for p in ids]
    append!(seeds, [path_seed(p, :sector, s) for p in ids for s in 1:11])
    append!(seeds, [path_seed(p, :name, j) for p in ids for j in 1:400])
    append!(seeds, [path_seed(p, :drift) for p in ids])
    @test allunique(seeds)
    @test_throws ArgumentError path_seed(1, :bogus)
    d = [Date(2014,1,2) + Day(i) for i in 0:20]
    @test growth_window(d, d[3], d[10]) == 3:9
    @test_throws ArgumentError growth_window(d, Date(2013,1,1), d[10])
end
```

(Add `using Random, Statistics` at the top of the test file.)

- [ ] **Step 2: Run to verify they fail**

Run: `julia --project=code code/test/test_synthetic_market.jl`
Expected: FAIL. `UndefVarError: drift_path`.

- [ ] **Step 3: Implement** the four functions in `SyntheticMarket.jl`, with docstrings that state units. Export them.

- [ ] **Step 4: Run to verify they pass**

Run: `julia --project=code code/test/test_synthetic_market.jl`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/src/SyntheticMarket.jl code/src/Include.jl code/test/test_synthetic_market.jl
git commit -m "feat(synthetic): path seeds, growth windows and the drift EMA"
```

---

### Task 3: Sector factors and leave-one-out loadings

**Files:**
- Modify: `code/src/SyntheticMarket.jl`, `code/src/Include.jl`, `code/test/test_synthetic_market.jl`

**Interfaces:**
- Consumes: `fit_base_sim`, `growth_rates`, `index_of` (existing), `growth_window` (Task 2).
- Produces:
  - `sim_residuals(md::MyMarketData, params::Dict{String,MySIMParameterEstimate}, start::Date, stop::Date)::Matrix{Float64}`:
    growth rows `growth_window(md.dates, start, stop)` × tickers in `md.tickers` order, with
    eᵢ = gᵢ − αᵢ − βᵢ·g_m, in yr⁻¹.
  - `sector_factors(E::AbstractMatrix, sector::Vector{String})::NamedTuple{(:sectors, :F)}`: `sectors` sorted unique,
    and `F[:, s]` the equal-weight mean of the columns in sector s.
  - `loo_loadings(E::AbstractMatrix, sector::Vector{String})::Vector{Float64}`: δᵢ is the OLS slope, with intercept, of
    `E[:, i]` on the mean of the other columns in its sector. Throws `ArgumentError` for a sector with one name.

- [ ] **Step 1: Write the failing tests**

```julia
@testset "sector factors and loadings" begin
    md = synthetic_md(ρ=0.5)
    fit = fit_base_sim(md, md.dates[1], md.dates[end])
    E = sim_residuals(md, fit.params, md.dates[1], md.dates[end])
    @test size(E) == (length(md.dates) - 1, 48)
    @test maximum(abs.(vec(mean(E; dims=1)))) < 1e-8            # OLS residuals have zero mean
    sf = sector_factors(E, md.sector)
    @test sf.sectors == ["S1", "S2", "S3", "S4"]
    @test sf.F[:, 1] ≈ vec(mean(E[:, 1:12]; dims=2))
    @test isapprox(median(loo_loadings(E, md.sector)), 1.0; atol=0.1)   # common factor, equal loadings
    rng = Xoshiro(3); U = randn(rng, 5000, 12)                          # no common factor
    @test abs(mean(loo_loadings(U, fill("A", 12)))) < 0.05              # full-mean slope would be ≈ 1
    @test_throws ArgumentError loo_loadings(U[:, 1:2], ["A", "B"])
end
```

- [ ] **Step 2: Run to verify they fail**

Run: `julia --project=code code/test/test_synthetic_market.jl`
Expected: FAIL. `UndefVarError: sim_residuals`.

- [ ] **Step 3: Implement** the three functions and export them.

- [ ] **Step 4: Run to verify they pass**

Run: `julia --project=code code/test/test_synthetic_market.jl`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/src/SyntheticMarket.jl code/src/Include.jl code/test/test_synthetic_market.jl
git commit -m "feat(synthetic): sector residual factors and leave-one-out loadings"
```

---

### Task 4: Composition with the variance correction

**Files:**
- Modify: `code/src/SyntheticMarket.jl`, `code/src/Include.jl`, `code/test/test_synthetic_market.jl`

**Interfaces:**
- Produces:
  - `const SYNTH_FLOOR = 0.10` and `const SYNTH_R2_PRESERVE = 0.80`.
  - `compose_name(α::AbstractVector, β::AbstractVector, δ::Real, g_m::AbstractVector, f::AbstractVector, ε::AbstractVector, σ²_m::Real, σ²_s::Real, σ²_gen::Real, r²::Real; floor=SYNTH_FLOOR, r2_threshold=SYNTH_R2_PRESERVE)::NamedTuple{(:g, :β_eff, :n_clipped)}`.
    α and β are per-day paths (yr⁻¹ and dimensionless). ε is the raw JumpHMM draw, demeaned inside the function.
    `g` is yr⁻¹. `β_eff` is the per-day β after clipping. `n_clipped` is the count of clipped days.

- [ ] **Step 1: Write the failing tests** (n = 200_000 Gaussian days; σ²_m = 6.25, σ²_s = 1.0, σ²_gen = 16.0;
  `f` ~ N(0, 1); `ε` ~ N(0, 4²); `g_m` ~ N(0, 2.5²); all from `Xoshiro(11)`)

```julia
@testset "composition" begin
    # (setup as above; olsβ(y, x) = cov(y, x) / var(x))
    c = compose_name(fill(0.0, n), fill(1.2, n), 0.8, gm, f, ε, 6.25, 1.0, 16.0, 0.3)
    @test isapprox(var(c.g), 16.0; rtol=0.02) && isapprox(olsβ(c.g, gm), 1.2; atol=0.02) && c.n_clipped == 0
    for b in (3.0, -3.0)                                    # ρ = (9·6.25 + 0.64)/16 > 0.9: clipped, sign kept
        c = compose_name(fill(0.0, n), fill(b, n), 0.8, gm, f, ε, 6.25, 1.0, 16.0, 0.3)
        ρ = (9 * 6.25 + 0.64) / 16
        @test all(c.β_eff .≈ b * sqrt(0.9 / ρ)) && c.n_clipped == n && isapprox(var(c.g), 16.0; rtol=0.03)
    end
    βt = repeat([0.5, 1.5], inner=1000, outer=n ÷ 2000)    # drifting β
    c = compose_name(fill(0.0, n), βt, 0.8, gm, f, ε, 6.25, 1.0, 16.0, 0.3)
    @test isapprox(var(c.g), 16.0; rtol=0.03) && mean(c.β_eff) ≈ mean(βt)
    c = compose_name(fill(0.0, n), fill(1.0, n), 0.0, gm, f, ε, 6.25, 1.0, 16.0, 0.9)   # R² branch
    @test isapprox(var(c.g), 6.25 / 0.9; rtol=0.03)
    c = compose_name(fill(0.3, n), fill(1.0, n), 0.8, gm, f, ε .+ 5.0, 6.25, 1.0, 16.0, 0.3)
    @test isapprox(mean(c.g), 0.3 + mean(gm) + 0.8 * mean(f); atol=1e-10)              # ε demeaned, α passes through
end
```

- [ ] **Step 2: Run to verify they fail**

Run: `julia --project=code code/test/test_synthetic_market.jl`
Expected: FAIL. `UndefVarError: compose_name`.

- [ ] **Step 3: Implement `compose_name`.** Set ε̃ = ε − mean(ε). Then for each day t, let
  v = β[t]²σ²_m + δ²σ²_s and ρ = v/σ²_gen:
  - If r² ≥ r2_threshold: set s² = v(1 − r²)/r²/σ²_gen and keep β[t] and δ.
  - Else, if ρ ≤ 1 − floor: set s² = 1 − ρ and keep β[t] and δ.
  - Otherwise: set c = √((1 − floor)/ρ), use c·β[t] and c·δ, set s² = floor, and count the day as clipped.

  Then g[t] = α[t] + b·g_m[t] + d·f[t] + √s²·ε̃[t], where b and d are the β and δ chosen above. This is the paper's
  `compose_hybrid` (`~/Desktop/papers/hybrid-jump-hmm-paper/code/src/Composers.jl:66`), with β²σ²_m replaced by v
  and evaluated per day.

- [ ] **Step 4: Run to verify they pass**

Run: `julia --project=code code/test/test_synthetic_market.jl`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/src/SyntheticMarket.jl code/src/Include.jl code/test/test_synthetic_market.jl
git commit -m "feat(synthetic): variance-corrected composition with sector term and per-day clipping"
```

---

### Task 5: Generator type, path draws and the MyMarketData builder

**Files:**
- Modify: `code/src/SyntheticMarket.jl`, `code/src/Include.jl`, `code/test/helpers.jl`, `code/test/test_synthetic_market.jl`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces:
  - `mutable struct MySyntheticGenerator`, built with `build(MySyntheticGenerator, nt)`. It throws `ArgumentError` on
    a missing field. Fields:
    - `dates::Vector{Date}`, `tickers::Vector{String}`, `sector::Vector{String}`, `sectors::Vector{String}` (sorted
      unique), `sector_index::Vector{Int}`;
    - `market`, `names::Vector`, `factors::Vector`: JumpHMM models for the market, each name and each sector;
    - `α_center::Vector{Float64}` (yr⁻¹), `β_center::Vector{Float64}`, `δ::Vector{Float64}`, `r²::Vector{Float64}`;
    - `σ²_m::Float64`, `σ²_s::Vector{Float64}`, `σ²_gen::Vector{Float64}` (yr⁻²);
    - `h_β::Float64`, `k_β::Vector{Float64}`, `h_α::Float64`, `k_α::Float64`, `φ::Float64`;
    - `close0::Vector{Float64}`, `spy0::Float64` (USD/share), `volume::Vector{Float64}` (shares/day),
      `g_f::Vector{Float64}` (yr⁻¹, one per date).
  - `with_generator(gen; kw...)::MySyntheticGenerator`: a copy with the named fields replaced.
  - `synthetic_draws(gen, p::Int)::NamedTuple{(:g_m, :jumps, :F, :E, :Zβ, :Zα)}`, with n = `length(gen.dates) − 1`:
    - `g_m`, `jumps` come from `JumpHMM.simulate(gen.market, n; n_paths=1, seed=path_seed(p, :market))`;
    - `F` is n × S, column s seeded `path_seed(p, :sector, s)`. It is not demeaned: the real factor's mean over
      2014–2024 is 0 by construction, and a draw's own drift is part of the scenario, a sector rotation (my call);
    - `E` is n × N, column j seeded `path_seed(p, :name, j)`, threaded over j;
    - `Zβ` and `Zα` are n × N standard normals from `Xoshiro(path_seed(p, :drift))`: all of `Zβ` column by column,
      then all of `Zα`.
  - `synthetic_growth(gen, d)::NamedTuple{(:G, :β_eff, :clip_share)}`, where for each name i:
    - αᵢ(t) = `α_center[i] + drift_path(Zα[:, i], h_α, k_α)`;
    - βᵢ(t) = `β_center[i] + drift_path(Zβ[:, i], h_β, k_β[i])`;
    - `compose_name` is called with `F[:, sector_index[i]]`, `σ²_s[sector_index[i]]`, `σ²_gen[i]` and `r²[i]`;
    - `clip_share` is the clipped share of name-days.
  - `market_data_from_growth(gen, g_m::Vector, G::Matrix)::MyMarketData`:
    - closes start at `close0` and `spy0` and compound with exp(g·Δt);
    - `open = close`, `spy_open = spy_close`;
    - volume is `gen.volume` held constant on every date;
    - `sector`, `g_f` and `dates` come from `gen`.

    It throws `ArgumentError` naming the first ticker with a close that is not finite and > 0.
  - `synthetic_market(gen, p::Int)::NamedTuple{(:md, :draws, :β_eff, :clip_share)}`.
- Test helper (`helpers.jl`): `toy_generator()` built from `synthetic_md(ρ=0.3)`.
  - The market, the names and the four sector factors are fit with `JumpHMM.fit(...; N=10)` and no tune.
  - Centers and r² come from `fit_base_sim`, δ from `loo_loadings`, and the variances from the toy's own data.
  - Drift: `h_β = 126.0`, `k_β = fill(2.0, 48)`, `h_α = 252.0`, `k_α = 0.5`, `φ = 1.0`.

- [ ] **Step 1: Write the failing tests**

```julia
@testset "synthetic market path" begin
    gen = toy_generator(); n = length(gen.dates) - 1
    a = synthetic_market(gen, 7); b = synthetic_market(gen, 7); c = synthetic_market(gen, 8)
    for fld in fieldnames(MyMarketData)                                   # V7: bitwise
        @test getfield(a.md, fld) == getfield(b.md, fld)
    end
    @test a.md.close != c.md.close
    J = VLINFORMS2026.JumpHMM
    @test a.draws.g_m == J.simulate(gen.market, n; n_paths=1, seed=path_seed(7, :market)).paths[1].observations
    for j in (1, 48)                                                      # independent of threading
        @test a.draws.E[:, j] == J.simulate(gen.names[j], n; n_paths=1, seed=path_seed(7, :name, j)).paths[1].observations
    end
    @test size(a.md.close) == (n + 1, 48) && a.md.open == a.md.close
    @test growth_rates(a.md.close) ≈ synthetic_growth(gen, a.draws).G
    @test growth_rates(a.md.spy_close) ≈ a.draws.g_m
    g0 = with_generator(gen; k_β=zeros(48), k_α=0.0)
    s0 = synthetic_growth(g0, a.draws)                                    # toy β ≈ 1 never clips
    @test s0.clip_share == 0.0 && s0.β_eff == repeat(gen.β_center', n)
    G = synthetic_growth(gen, a.draws).G; G[5, 3] = -Inf
    @test_throws ArgumentError market_data_from_growth(gen, a.draws.g_m, G)
end
```

- [ ] **Step 2: Run to verify they fail**

Run: `julia --project=code code/test/test_synthetic_market.jl`
Expected: FAIL. `UndefVarError: toy_generator` / `synthetic_market`.

- [ ] **Step 3: Implement** the type, the factory, `with_generator`, the four functions and `toy_generator`. Export
  everything except `toy_generator`.

- [ ] **Step 4: Run to verify they pass**

Run: `julia --project=code -t 4 code/test/test_synthetic_market.jl`
Expected: PASS. Run once with `-t 1` too: same result.

- [ ] **Step 5: Commit**

```bash
git add code/src/SyntheticMarket.jl code/src/Include.jl code/test/helpers.jl code/test/test_synthetic_market.jl
git commit -m "feat(synthetic): generator type, seeded path draws and MyMarketData builder"
```

---

### Task 6: Fitting the generator, and script 20

**Files:**
- Create: `code/src/SyntheticCalibration.jl`, `code/test/test_synthetic_calibration.jl`, `scripts/20_synthetic_fit.jl`
- Modify: `code/src/Include.jl`, `code/test/runtests.jl`

**Interfaces:**
- Consumes: Tasks 2–5, `fit_base_sim`, `estimate_sim`, `ewls_init`, `ewls_update!`, `online_sim`, `load_calibration`.
- Produces:
  - `const SYNTH_HMM = (N=50, ν=5.0, rf=0.0, dt=1/252, ϵ_range=range(1e-4, 2.5e-2, length=10), λ_range=range(10.0, 120.0, length=10), n_paths=100)`.
  - `fit_name_model(prices::AbstractVector{<:Real}; seed::Int, N::Int=SYNTH_HMM.N, tune::Bool=true)`:
    `JumpHMM.fit` then `JumpHMM.tune(...; seed=seed)` with the `SYNTH_HMM` settings.
  - `model_sd_ratio(model, g_real::AbstractVector; seed::Int, n_steps::Int=100_000)::Float64`: the SD of a long draw
    divided by `std(g_real)`.
  - `ewls_beta_path(g_i, g_m, prior::MySIMParameterEstimate, m_m, v_m; half_life=63.0, prior_weight=63.0)::Vector{Float64}`:
    β̂ after each update, the same recursion as `online_sim`.
  - `beta_path_stats(b::AbstractVector)::NamedTuple{(:sd, :acf252)}`: `std(b)` and `cor(b[1:end-252], b[253:end])`.
  - `real_drift_targets(md, calib)::NamedTuple{(:sd_β, :acf252, :α_full, :α_h1, :α_h2, :spread, :corr)}`:
    - `sd_β` per name and `acf252` (median over names) come from `online_sim(md, calib; start=2014-01-06, stop=2024-12-31)`;
    - the α values are OLS α from `fit_base_sim` on 2014-01-03 → 2024-12-31, 2014-01-03 → 2019-12-31 and
      2020-01-02 → 2024-12-31;
    - `spread = std(α_full)` and `corr = cor(α_h1, α_h2)`.
  - `const SYNTH_TRAIN = (Date(2014,1,3), Date(2024,12,31))` and
    `const SYNTH_ALPHA_WINDOWS = ((Date(2014,1,3), Date(2024,12,31)), (Date(2014,1,3), Date(2019,12,31)), (Date(2020,1,2), Date(2024,12,31)))`.
    These are the defaults of the `train` and `windows` keywords below; tests pass toy windows.
  - `bank_beta_stats(gen, bank::Vector; train=SYNTH_TRAIN)::NamedTuple{(:sd_β, :acf252)}`, per name (`sd_β`) and the
    median over names (`acf252`). For each name and bank path:
    - compose the name;
    - fit its prior with `estimate_sim` on the training window;
    - take `m_m` and `v_m` from the path's g_m (mean, population variance);
    - run `ewls_beta_path` over the training rows;
    - take `beta_path_stats`.

    `sd_β` and each name's acf252 are means over the bank. The field names match `real_drift_targets`, so the output
    can serve as a target.
  - `fit_beta_drift(gen0, bank::Vector, targets; h_grid=[63.0, 126.0, 252.0, 504.0, 1008.0, 2016.0], sd_max=1.5, train=SYNTH_TRAIN)::NamedTuple{(:h_β, :k_β, :table)}`.
  - `alpha_parts(gen1, d, i::Int; windows=SYNTH_ALPHA_WINDOWS)::NamedTuple{(:rest, :unit)}`: 3-vectors of name i's OLS
    α̂ in the windows (full, h1, h2), once with `α_center[i] = 0` and `k_α = 0`, and once for the unit drift path
    (`drift_path(Zα[:, i], h_α, 1.0)`) alone.
  - `fit_alpha_drift(gen1, bank::Vector, targets; φ_grid=0.0:0.01:1.5, sdα_grid=0.0:0.0025:0.15, windows=SYNTH_ALPHA_WINDOWS)::NamedTuple{(:φ, :k_α, :spread, :corr, :loss)}`.
  - Output files:
    - `results/synthetic/generator.jld2`, key `"generator"`;
    - `results/synthetic/generator_summary.toml`.

- [ ] **Step 1: Write the failing tests** in `test_synthetic_calibration.jl`. Use `toy_generator()`; bank ids are
  `1_000_001:1_000_010` for the truth and `1_000_011:1_000_020` for the fit.

```julia
@testset "ewls β path matches online_sim" begin
    md = synthetic_md(ρ=0.3); f = fit_base_sim(md, md.dates[1], md.dates[800])
    c = build(MyCalibration, (base_sim=f.params, σ_m=f.σ_m, m_m=f.m_m, v_m=f.v_m, λ=0.5,
        train_start=md.dates[1], train_stop=md.dates[800]))
    sim = online_sim(md, c; start=md.dates[2], stop=md.dates[800])
    G = growth_rates(md.close); gm = growth_rates(md.spy_close)
    @test ewls_beta_path(G[1:799, 5], gm[1:799], f.params[md.tickers[5]], f.m_m, f.v_m) ≈ sim.β[:, 5]
end
@testset "β drift recovery" begin
    gen = toy_generator(); a = ema_rate(126); kt = 0.3 / sqrt(a / (2 - a))
    tr = (gen.dates[1], gen.dates[end])
    truth = with_generator(gen; h_β=126.0, k_β=fill(kt, 48), k_α=0.0)
    tg = bank_beta_stats(truth, [synthetic_draws(truth, p) for p in 1_000_001:1_000_010]; train=tr)
    fit = fit_beta_drift(with_generator(gen; k_β=zeros(48), k_α=0.0),
        [synthetic_draws(gen, p) for p in 1_000_011:1_000_020], tg; h_grid=[63.0, 126.0, 252.0], train=tr)
    @test fit.h_β == 126.0 && isapprox(median(fit.k_β), kt; rtol=0.15)
    low = (sd_β=fill(1e-6, 48), acf252=tg.acf252)
    @test all(iszero, fit_beta_drift(with_generator(gen; k_β=zeros(48), k_α=0.0),
        [synthetic_draws(gen, p) for p in 1_000_011:1_000_012], low; h_grid=[126.0], train=tr).k_β)  # at or below noise → k = 0
end
@testset "α decomposition is exact, and φ is recovered" begin
    gen = toy_generator(); d = synthetic_draws(gen, 1_000_011); i = 9
    wins = ((gen.dates[1], gen.dates[end]), (gen.dates[1], gen.dates[700]), (gen.dates[701], gen.dates[end]))
    g1 = with_generator(gen; α_center=fill(0.07, 48), k_α=0.8)
    G = synthetic_growth(g1, d).G; w = growth_window(gen.dates, wins[1]...)
    p = alpha_parts(gen, d, i; windows=wins)
    @test estimate_sim(d.g_m[w], G[w, i]).α ≈ 0.07 + p.rest[1] + 0.8 * p.unit[1] atol=1e-8
    # recovery: truth φ = 0.55 on centers c = ā + φ(α̂ − ā); fit on a separate bank → |φ̂ − 0.55| ≤ 0.10
end
@testset "model draw SD gate" begin
    md = synthetic_md(ρ=0.3); m = fit_name_model(md.close[:, 1]; seed=1, N=10, tune=false)
    @test 0.7 < model_sd_ratio(m, growth_rates(md.close[:, 1]); seed=1) < 1.3
end
```

Write the α recovery test in full, in the same style as the β one. Add the file to `runtests.jl`.

- [ ] **Step 2: Run to verify they fail**

Run: `julia --project=code -t 8 code/test/test_synthetic_calibration.jl`
Expected: FAIL. `UndefVarError: ewls_beta_path`.

- [ ] **Step 3: Implement `SyntheticCalibration.jl`.** Two algorithms are not fixed by the signatures above.

  **`fit_beta_drift`.** α is held at `α_center` with `k_α = 0`. For each h in `h_grid`, and for each name (threaded):
  1. Bisect on the stationary SD σₓ ∈ [0, sd_max], 30 halvings, with k = σₓ/√(a/(2−a)).
  2. At each candidate, compute the name's bank-mean SD the way `bank_beta_stats` does.
  3. If the SD at k = 0 already reaches the target, set k = 0. The bank is reused, so these are common random numbers.

  Then pick the h whose median over names of the bank-mean `acf252` is closest to `targets.acf252`. `table` has one
  row per h: h, median k, share with k = 0, median synthetic acf252.

  **`fit_alpha_drift`.** OLS is linear in gᵢ, and α does not enter the clipping, so for each window w:
  α̂ᵢ,w = cᵢ(φ) + restᵢ,w + k·unitᵢ,w, where cᵢ(φ) = ā + φ(α̂ᵢ,real − ā) and ā = `mean(targets.α_full)`.
  1. Precompute `alpha_parts` for every bank path and name once.
  2. For every grid point (φ, σ_α), with k = σ_α/√(a/(2−a)) at h_α = 252: per path, take the spread
     `std(α̂_full)` and the corr `cor(α̂_h1, α̂_h2)`, then average over paths.
  3. The loss is ((spread − targets.spread)/0.01)² + ((corr − targets.corr)/0.05)². Return the minimizer, and set
     `α_center` = c(φ̂) in the generator.

- [ ] **Step 4: Run to verify they pass**

Run: `julia --project=code -t 8 code/test/test_synthetic_calibration.jl`
Expected: PASS.

- [ ] **Step 5: Write `scripts/20_synthetic_fit.jl`.** Activate `code` the way `scripts/11_run_study.jl` does. Steps:
  1. Load the real data: `load_market_data(DATA_DIR; start=2014-01-03, stop=2026-09-04)`. Load
     `results/calibration.jld2` and assert its train window is 2014-01-03 → 2024-12-31.
  2. Refit the 373 name models on the training closes, threaded, each with `seed = 5660 + j`. Compute
     `model_sd_ratio` per name.
     - **Gate:** if any ratio is outside [0.5, 2.0], stop with an error naming those tickers.
     - Print the ratios for BF.B, CHD, CTSH, EOG, GOOG, MA, PPG, TECH and UNP.
  3. Compute the residuals, the sector factors and δ (`loo_loadings`). σ²_m, σ²_s and σ²_gen are the real
     training-window sample variances. r² comes from `calib.base_sim`.
  4. Fit the 11 sector models on `exp.(vcat(0, cumsum(F[:, s] .* Δt)))`, with `seed = 7000 + s`.
  5. Load the market model from the eCornell surrogate file, key `"model"`.
  6. Build `gen0`:
     - `α_center` = real OLS α and `β_center` = real OLS β;
     - `k_β = 0`, `k_α = 0`, `φ = 1`, `h_α = 252`;
     - `close0` = the real closes on 2014-01-03 and `spy0` the real SPY close;
     - `volume` = the median real share volume over the training window;
     - `g_f` = `md.g_f`.
  7. Draw the bank, ids `1_000_001:1_000_020`. Run `fit_beta_drift`, then `fit_alpha_drift` on the β-fitted generator.
     **If k_α comes out 0, stop and report to the user:** decision 4 says α drifts, and the data would leave no room
     for it.
  8. Save `generator.jld2`. Write the summary TOML:
     - the JumpHMM rev;
     - the per-h table, h_β, the median and quartiles of k_β, the share with k_β = 0;
     - φ, k_α and the stationary SD of α;
     - the fitted spread and corr against the targets;
     - for reference, the spec's pooled α statistic, mean(Δᵢ²) − mean(noise), on real data and on the fitted bank;
     - the real targets: check they print about 0.076 and 0.18;
     - the δ median per sector and σ_s per sector;
     - the model SD ratio median and range;
     - the wall time.

- [ ] **Step 6: Run script 20**

Run: `julia --project=code -t 30 scripts/20_synthetic_fit.jl`
Expected: finishes in under 15 minutes, no gate errors, and the real targets print near 0.076 and 0.18.

- [ ] **Step 7: Commit**

```bash
git add code/src/SyntheticCalibration.jl code/src/Include.jl code/test/test_synthetic_calibration.jl code/test/runtests.jl scripts/20_synthetic_fit.jl results/synthetic/generator_summary.toml
git commit -m "feat(synthetic): refit name and sector JumpHMMs, fit β and α drift (script 20)"
```

---

### Task 7: Market paths and the three sets (script 21)

**Files:**
- Create: `code/src/SyntheticRunner.jl`, `code/test/test_synthetic_runner.jl`, `scripts/21_synthetic_paths.jl`
- Modify: `code/src/Include.jl`, `code/test/runtests.jl`

**Interfaces:**
- Consumes: `MySyntheticGenerator`, `path_seed`, `growth_window`.
- Produces:
  - `market_path_stats(gen, ids::AbstractVector{Int}; window=(Date(2025,1,2), Date(2026,9,4)), train=(Date(2014,1,3), Date(2024,12,31)))::DataFrame`.
    It is threaded over ids and draws the market exactly as `synthetic_draws` does. Columns:
    - `path_id`;
    - `window_max_drawdown`: the fraction, on synthetic SPY closes within the window, with the peak reset at the
      window start;
    - `window_growth` and `train_growth` (yr⁻¹);
    - `has_burst`: any jump flag on the window's growth rows.
  - `path_sets(stats::DataFrame; n_burst=200, n_bear=200, n_typical=600, seed=5660)::DataFrame`. It adds:
    - `burst_stress`: the n_burst deepest among `has_burst`;
    - `bear_stress`: the n_bear deepest among `!has_burst`;
    - `typical_rank`: the position in `randperm(MersenneTwister(seed), nrow)`;
    - `typical = typical_rank ≤ n_typical`;
    - `run_order`: missing for paths in no set; otherwise sorted by (priority, path_id), where priority is the
      smallest rank-in-set ÷ set-size over the path's sets. Stopping early then leaves every set equally complete.

    It throws `ArgumentError` if a stress pool is smaller than its quota.

- [ ] **Step 1: Write the failing tests**

```julia
@testset "market path stats" begin
    gen = toy_generator(); st = market_path_stats(gen, 1:5; window=(gen.dates[1001], gen.dates[1400]), train=(gen.dates[1], gen.dates[1000]))
    md = synthetic_market(gen, 3).md; i0, i1 = 1001, 1400
    S = md.spy_close[i0:i1]
    @test st.window_max_drawdown[3] ≈ maximum(1 .- S ./ accumulate(max, S))
    @test st.train_growth[3] ≈ log(md.spy_close[1000] / md.spy_close[1]) / (999 / 252)
    @test !any(st.has_burst)                                     # toy market is untuned: ϵ = 0
end
@testset "path sets" begin
    df = DataFrame(path_id=1:1000, window_max_drawdown=rand(Xoshiro(2), 1000), has_burst=(1:1000) .% 10 .== 0)
    ps = path_sets(df; n_burst=5, n_bear=5, n_typical=10)
    b = df[df.has_burst, :]; @test sort(ps.path_id[ps.burst_stress]) == sort(b.path_id[sortperm(b.window_max_drawdown; rev=true)[1:5]])
    @test count(ps.bear_stress) == 5 && count(ps.typical) == 10 && !any(ps.bear_stress .& ps.has_burst)
    ro = collect(skipmissing(ps.run_order)); @test sort(ro) == 1:length(ro)
    first4 = ps[coalesce.(ps.run_order, 10^9) .<= 4, :]
    @test any(first4.burst_stress) && any(first4.bear_stress) && any(first4.typical)
    @test_throws ArgumentError path_sets(df; n_burst=500)
end
```

- [ ] **Step 2: Run to verify they fail**

Run: `julia --project=code code/test/test_synthetic_runner.jl`
Expected: FAIL. `UndefVarError: market_path_stats`.

- [ ] **Step 3: Implement both functions.** Export them.

- [ ] **Step 4: Run to verify they pass**

Run: `julia --project=code code/test/test_synthetic_runner.jl`
Expected: PASS.

- [ ] **Step 5: Write and run `scripts/21_synthetic_paths.jl`.** It loads the generator, runs
  `market_path_stats(gen, 1:10_000)` and `path_sets`, and writes `results/synthetic/paths.csv`. It prints:
  - the share of windows with a burst (the builder check says about 5.4%);
  - the share of the 50 deepest drawdowns with a burst (about 68%);
  - the median `train_growth` against real SPY over 2014–2024, a V1 preview;
  - the drawdown range of each set.

Run: `julia --project=code -t 30 scripts/21_synthetic_paths.jl`
Expected: 10,000 rows; sets of 200, 200 and 600.

- [ ] **Step 6: Commit**

```bash
git add code/src/SyntheticRunner.jl code/src/Include.jl code/test/test_synthetic_runner.jl code/test/runtests.jl scripts/21_synthetic_paths.jl results/synthetic/paths.csv
git commit -m "feat(synthetic): 10,000 market paths ranked into burst, bear and typical sets (script 21)"
```

---

### Task 8: Checks V1–V9 (script 24): the Tuesday gate

**Files:**
- Create: `code/src/SyntheticChecks.jl`, `code/test/test_synthetic_checks.jl`, `scripts/24_synthetic_checks.jl`
- Modify: `code/src/Include.jl`, `code/test/runtests.jl`

**Interfaces:**
- Consumes: Tasks 2–7, `calibrate`, `online_sim`, `estimate_sim`.
- Produces:
  - `ew_rebalance_gap(G::AbstractMatrix; Δt=1/252)::Float64` (yr⁻¹): the daily-rebalanced minus the buy-and-hold
    equal-weight growth over the rows of G.
    - Rebalanced: Σₜ log(meanᵢ exp(G[t,i]Δt)).
    - Buy-and-hold: log(meanᵢ exp(Σₜ G[t,i]Δt)).
    - Both are divided by (rows·Δt).
  - `abs_acf(x::AbstractVector, lags)::Vector{Float64}`: `cor(abs.(x[1:end-L]), abs.(x[1+L:end]))` for each lag L.
  - `mean_pairwise_corr(R::AbstractMatrix)::Float64`: the mean of the off-diagonal entries of `cor(R)`.
  - `real_reference(md_real, calib_real, gen)::NamedTuple`: everything real that the checks compare against.
    - the training-window growth G and g_m;
    - the per-sector and cross-sector mean residual correlations;
    - `sd_β` and `acf252` from `real_drift_targets`;
    - the per-name abs-growth ACFs at lags 1, 5 and 20;
    - the real `ew_rebalance_gap`, `spread`, `corr` and SPY training growth;
    - `n_beta_le0` from the real calibration's `online_sim` over the trading window.
  - `path_diagnostics(gen, sm, calib_syn, ref)::NamedTuple`, on the training rows unless noted. `ref` holds the real
    G, g_m, sector residual correlations, `sd_β` targets and ACFs. Fields:
    - V2: `ew_gap`;
    - V3: `var_ratio_draw` (var gᵢ ÷ var of the raw draw `E[:, i]`), `var_ratio_real` (÷ σ²_gen), and `β_diff`
      (OLS β − mean β_eff);
    - V4: `acf_syn`, `acf_draw`, `acf_real` (N × 3, lags 1, 5, 20);
    - V5: `sector_corr` (S) and `cross_corr`;
    - V6: `ewls_sd` (N), `acf252`, and `n_beta_le0`, the names whose online β̂ is ≤ 0 on any trading-window day,
      from `online_sim(md, calib_syn; start=2025-01-02, stop=2026-09-04)`;
    - V9: `α_spread` and `α_corr`;
    - `clip_share`.
  - `check_table(diags::Vector, ref, v1::NamedTuple)::DataFrame`, with columns `check, statistic, synthetic,
    reference, threshold, pass` (`pass` is `missing` for report-only rows). The gates:

    | Check | Gate |
    |---|---|
    | V1 | \|median train growth − real\| ≤ 0.02 |
    | V2 | \|median ew_gap − real gap\| ≤ 0.02 |
    | V3 | median `var_ratio_draw` within [0.95, 1.05]; median \|β_diff\| ≤ 0.05 |
    | V4 | \|median(acf_syn − acf_draw)\| ≤ 0.05 at each lag |
    | V5 | \|median sector_corr − real\| ≤ 0.05 for every sector |
    | V6 | median over names of (median-over-paths `ewls_sd` ÷ `sd_β`) within [0.9, 1.1] |
    | V9 | \|median α_spread − ref.spread\| ≤ 0.01 and \|median α_corr − ref.corr\| ≤ 0.05 |

    V3 gates on the name's own draw, not on σ²_gen. This follows §8: "V3 and V4 gate on the composition, not the
    models". The stored models ran about 7% above real variance, so a gate on σ²_gen would fail on the models alone.
    Reported but not gated: V3 `var_ratio_real`; V4 against real; V5 across sectors; V6 `acf252` and the share of
    paths with `n_beta_le0 ≥ 1`, next to the real count.

- [ ] **Step 1: Write the failing tests**

```julia
@testset "check helpers" begin
    G = [1.0 3.0; -2.0 0.5; 4.0 -1.0]; Δt = 1/252
    reb = sum(log(mean(exp.(G[t, :] .* Δt))) for t in 1:3); bh = log(mean(exp.(vec(sum(G; dims=1)) .* Δt)))
    @test ew_rebalance_gap(G) ≈ (reb - bh) / (3Δt)
    @test ew_rebalance_gap(hcat(G[:, 1], G[:, 1])) ≈ 0.0 atol=1e-12
    x = randn(Xoshiro(4), 1000); @test abs_acf(x, [1, 5]) ≈ [cor(abs.(x[1:end-1]), abs.(x[2:end])), cor(abs.(x[1:end-5]), abs.(x[6:end]))]
    R = randn(Xoshiro(5), 500, 3); C = cor(R); @test mean_pairwise_corr(R) ≈ (C[1,2] + C[1,3] + C[2,3]) / 3
end
@testset "check gates" begin
    # check_table on hand-made diags: V1 diff 0.019 passes and 0.021 fails; V3 median var ratio 1.06 fails;
    # report-only rows have pass === missing
end
```

Write the gate testset in full: build two diags vectors that differ only in the field under test, and assert the
`pass` value of that row.

- [ ] **Step 2: Run to verify they fail**

Run: `julia --project=code code/test/test_synthetic_checks.jl`
Expected: FAIL. `UndefVarError: ew_rebalance_gap`.

- [ ] **Step 3: Implement** `SyntheticChecks.jl` and export its functions.

- [ ] **Step 4: Run to verify they pass**

Run: `julia --project=code code/test/test_synthetic_checks.jl`
Expected: PASS.

- [ ] **Step 5: Write `scripts/24_synthetic_checks.jl`.** It:
  - takes the check paths: the first 50 by `typical_rank` from `paths.csv`;
  - for each one, runs `synthetic_market`, then `calibrate(md; train_start, train_stop, window=s.omega_window)`, then
    `path_diagnostics`;
  - computes V1 from `paths.csv` over all 10,000 paths;
  - checks V7 by building path `first(ids)` twice and comparing every field with `==`;
  - writes `results/synthetic/checks.md` (a header plus `markdown_table`) and `checks.csv`;
  - exits 1 if any gate fails.

- [ ] **Step 6: Run the checks and the full suite (V8)**

Run: `julia --project=code -t 30 scripts/24_synthetic_checks.jl`, then `julia --project=code code/test/runtests.jl`
Expected: every gate passes; the suite exits 0, with the Task 1 baseline count plus the new testsets.
**If any gate fails: stop. Report the check table to the user in plain words and do not tune.** This is the Tuesday
gate; the runner is not started until it passes or the user decides.

- [ ] **Step 7: Commit**

```bash
git add code/src/SyntheticChecks.jl code/src/Include.jl code/test/test_synthetic_checks.jl code/test/runtests.jl scripts/24_synthetic_checks.jl results/synthetic/checks.md results/synthetic/checks.csv
git commit -m "feat(synthetic): checks V1-V9 on 50 typical paths (script 24)"
```

---

### Task 9: Per-path study runner and the 20-path smoke run (script 22)

**Files:**
- Modify: `code/src/StudyRunner.jl` (factor out `_study_row`), `code/src/SyntheticRunner.jl`, `code/src/Include.jl`,
  `code/test/test_synthetic_runner.jl`
- Create: `scripts/22_synthetic_run.jl`

**Interfaces:**
- Consumes: `tier1_runs`, `universe_runs`, `fixed_list_runs`, `with_settings`, `context_key`, `build_context`,
  `run_study`, `growth_scorecard`, `study_stats`, `regime_stats`, `study_hash`, `study_key`, `data_fingerprint`,
  `passive_references`, `calibrate`, `online_sim`, and `synthetic_market`, `path_sets` output.
- Produces:
  - `_study_row(key::NamedTuple, label, sc, st, wall)::Dict{Symbol,Any}` in `StudyRunner.jl`, with exactly the
    `_STUDY_SUMMARY_COLUMNS` keys except `:hash` and `:status`. `_study_summary_row` calls it, so its output is
    unchanged.
  - `synthetic_runs(s::MyStudySettings, grid; n_seeds::Int=5)::Vector{MyStudyRun}`: 532 runs.
    - `tier1_runs` with `n_seeds = 5`: 154 runs.
    - `universe_runs` at K = 13 with every universe screen: 182 runs.
    - `universe_runs` at K ∈ {5, 34} with `[:global_gamma, :random_global]`: 168 runs.
    - `fixed_list_runs`: 28 runs.
  - `run_synthetic_path(md, calib, runs)::DataFrame`, columns `_STUDY_SUMMARY_COLUMNS` then `:error`:
    - one context per distinct `context_key`;
    - runs threaded with `Threads.@threads :dynamic`;
    - each run scored exactly as `run_and_save_study` scores it, but no file is written;
    - a failure becomes a row with `status = "error"`, the message in `error`, and the other metrics missing;
    - rows are sorted by `hash`.
  - `path_done(dir, p)::Bool`: true when `path_<p>.csv` exists.
  - `write_path_files(dir, p, rows, bench, meta)`: writes `bench_<p>.csv` and `meta_<p>.csv`, then `path_<p>.csv`
    last. Each goes to `*.tmp` first and is then moved into place.

- [ ] **Step 1: Write the failing tests**

```julia
@testset "synthetic run list" begin
    s, grid = load_study(joinpath(@__DIR__, "..", "..", "config", "study.toml"))
    runs = synthetic_runs(s, grid)
    @test length(runs) == 532 && allunique(r.label for r in runs)
    @test all(r.seed in 5660:5664 for r in runs)
    @test count(r -> r.settings.basket_K in (5, 34) && r.screen in (:global_gamma, :random_global), runs) == 168
end
@testset "path rows match the cached runner" begin
    # study_md / study_run helpers: two runs, one CD/gamma and one random with seed 5661
    # rows = run_synthetic_path(md, calib, runs); run_study_grid(md, calib, runs; outdir=tmp)
    # ref = rebuild_study_summary(tmp, "t"): same column names in the same order (rows adds :error at the end)
    # and equal values in every column except wall_seconds
end
@testset "error rows and resumable writes" begin
    # a fixed-list run with ticker "NOPE" → that row has status == "error" and a non-empty error; the other rows are ok
    # path_done(tmp, 4) is false with only path_4.csv.tmp present, and true after write_path_files(tmp, 4, ...)
end
```

Write the two sketched testsets in full with the existing `study_md`/`study_run` helpers.

- [ ] **Step 2: Run to verify they fail**

Run: `julia --project=code -t 4 code/test/test_synthetic_runner.jl`
Expected: FAIL. `UndefVarError: synthetic_runs`.

- [ ] **Step 3: Implement.** Factor `_study_row` out of `_study_summary_row` first. Then run
  `julia --project=code code/test/test_study_runner.jl` and expect PASS unchanged. Then add the four functions.

- [ ] **Step 4: Run to verify they pass**

Run: `julia --project=code -t 4 code/test/test_synthetic_runner.jl`
Expected: PASS.

- [ ] **Step 5: Write `scripts/22_synthetic_run.jl`.** Arguments:
  - `--smoke`: the first 20 by `run_order`, written to `results/synthetic/smoke/`;
  - `--limit N`;
  - `--extra M`: also run typical ranks 601 … 600+M, by rank, after every set path.

  The default output is `results/synthetic/rows/`. Per path, skipping it if `path_done`:
  1. `synthetic_market(gen, p)`.
  2. `calibrate(md; train_start=s.train_start, train_stop=s.train_stop, window=s.omega_window)`. Log λ.
     - The real calibration took 3 s. If a path's calibration takes more than 30 s, switch the rest of the run to the
       fixed-λ calibration.
     - Fixed-λ calibration: `fit_base_sim` on the training window, then `build(MyCalibration, (...; λ=0.0))`.
     - Log `lambda_fixed = true` on every path that uses it.
  3. `run_synthetic_path(md, calib, synthetic_runs(s, grid))`. Add `path_id` and the set flags.
  4. `passive_references(md, s.start, s.stop; W0=s.W0)` gives the benchmark rows.
  5. The meta row: `path_id`, set flags, `lambda`, `lambda_fixed`, `gen_seconds`, `calib_seconds`, `grid_seconds`,
     `clip_share`, `n_beta_le0` (from `online_sim` over the trading window), `n_errors`, `threads`.
  6. `write_path_files`. Print the per-path wall time and the ETA.

- [ ] **Step 6: Smoke run**

Run: `julia --project=code -t 30 scripts/22_synthetic_run.jl --smoke`
Expected:
- 20 paths, 532 rows each, 0 error rows;
- the measured seconds per path printed (spec estimate about 40 s);
- a rerun prints 20 skipped;
- the median smoke `terminal / W0` for CD with γ per sector lies between 0.5 and 3. This is a sanity range, not a
  result.

If the measured time × 1,000 paths would not finish by Thu Oct 29 08:00 from a Wed 20:00 start, shrink the typical
set (`--limit`) and keep the two stress sets at 200 each. Tell the user the new count.

- [ ] **Step 7: Commit**

```bash
git add code/src/StudyRunner.jl code/src/SyntheticRunner.jl code/src/Include.jl code/test/test_synthetic_runner.jl scripts/22_synthetic_run.jl
git commit -m "feat(synthetic): per-path study runner with summary rows only, resumable (script 22)"
```

---

### Task 10: The overnight run (Wed Oct 28)

**Files:** none changed. Outputs go to `results/synthetic/rows/` (gitignored).

- [ ] **Step 1: Before launching, request one whole-branch review** (superpowers:requesting-code-review), covering
  Task 1's base to HEAD. Fix Critical and Important findings. Rerun the suite and script 24.

- [ ] **Step 2: Launch**

```bash
nohup julia --project=code -t 30 scripts/22_synthetic_run.jl > results/synthetic/run.log 2>&1 &
```

- [ ] **Step 3: Thursday morning: check progress**

Run: `ls results/synthetic/rows/path_*.csv | wc -l` and `grep -c error results/synthetic/rows/meta_*.csv`
Expected: about 1,000 paths and 0 errored paths. If the run finished early and time allows, relaunch with
`--extra M` to grow the typical set.

---

### Task 11: Tables (script 23)

**Files:**
- Create: `code/src/SyntheticTables.jl`, `code/test/test_synthetic_tables.jl`, `scripts/23_synthetic_tables.jl`
- Modify: `code/src/Include.jl`, `code/test/runtests.jl`

**Interfaces:**
- Consumes: the row, bench and meta CSVs from Task 9; `paths.csv`; `markdown_table`; `TABLE_ALLOCATOR_ORDER`.
- Produces:
  - `PICKER_ROWS`: an ordered vector of `(key, name)`. Keys are built by `picker_key`:
    - `"gamma"` → γ per sector (33); `"bandit"` → Bandit per sector (33); `"random"` → Random per sector (33);
    - `"global_gamma|K=5"`, `"…|K=13"`, `"…|K=34"` → Global γ (5), (13), (34);
    - `"corr_gamma|K=13"` → Correlation-aware γ (13); `"liquid_gamma|K=13"` → Liquid then γ (13);
    - `"random_global|K=5"`, `"…|K=13"`, `"…|K=34"` → Random, whole universe (5), (13), (34);
    - `"random_liquid|K=13"` → Random, liquid 100 (13);
    - `"fixed:l8b_13"` → User's 13 names (13); `"fixed:l8b_15"` → User's 13 + WMT, PG (15).
  - `picker_key(row)::String`, from `screen`, `basket_K` and the `fixed:` part of `label`.
  - `path_cells(rows::DataFrame)::DataFrame`: per (`path_id`, picker, `allocator`, `hedge`), the median over seeds of
    `wealth = terminal / W0`, `realized_growth`, `max_drawdown` and `sharpe`. Only `status == "ok"` rows count, and
    `n_seeds` records how many did.
  - `bench_cells(bench::DataFrame)::DataFrame`: per `path_id`, the same four metrics for "SPY buy-and-hold" and
    "equal-weight universe".
  - `grid_table(cells, bench, metric::Symbol; ids, hedge::Bool, stat::Symbol)::DataFrame`. Rows: the SPY and
    equal-weight rows, then `PICKER_ROWS`. Columns: `Ticker picker`, then the 7 utilities. `stat` is one of:
    - `:median`: the median across `ids`;
    - `:band`: the string `"m (p10–p90)"`, with 3 decimals;
    - `:beat_spy`: the share of ids whose cell beats that path's SPY value of `metric` (higher is better, except
      `max_drawdown`, where lower is better).

    In the `:median` wealth table, bold a cell whose `:beat_spy` share is above 0.5.
  - `head_to_head(cells; ids)::DataFrame`: per utility and hedge setting, per-path differences summarized as median,
    p10, p90 and the share > 0:
    - Global γ minus Random, whole universe, at K = 5, 13 and 34 (wealth);
    - γ per sector minus Random per sector (wealth);
    - hedge on minus off for `realized_growth` and `max_drawdown`, for each picker.
  - `real_percentiles(cells, real_cells; ids)::DataFrame`: the picker × utility grid of the share of typical paths
    whose wealth is at or below the real 2025–26 cell. Real cells are the median over seeds 5660–5664 of
    `results/tier1_summary.csv` and `results/universe_summary.csv`, the same 5 seeds as the synthetic runs.

- [ ] **Step 1: Write the failing tests** on a hand-made rows frame: 3 paths × 2 seeds × the CD and CES utilities ×
  the random per sector picker and the γ per sector picker, plus a bench frame.

```julia
@testset "synthetic tables" begin
    # path_cells: the median over 2 seeds equals the hand value for path 2, random per sector, CD
    # grid_table(:median): the first two rows are "SPY buy-and-hold" and "equal-weight universe"; picker rows follow
    #   PICKER_ROWS order; the CD cell equals the median over paths of the per-path medians
    # grid_table(:beat_spy): equals 2/3 when 2 of the 3 paths beat their own SPY
    # grid_table(:band): the string is "1.200 (1.040–1.360)" for a hand-picked case
    # an error row is excluded and n_seeds drops to 1 for that cell
end
```

Write it in full with concrete numbers.

- [ ] **Step 2: Run to verify they fail**

Run: `julia --project=code code/test/test_synthetic_tables.jl`
Expected: FAIL. `UndefVarError: path_cells`.

- [ ] **Step 3: Implement** `SyntheticTables.jl` and export its functions.

- [ ] **Step 4: Run to verify they pass**

Run: `julia --project=code code/test/test_synthetic_tables.jl`
Expected: PASS.

- [ ] **Step 5: Write `scripts/23_synthetic_tables.jl`.** It reads a rows directory (an argument; defaults to
  `results/synthetic/rows/`) and writes `results/synthetic/tables.md` and `results/synthetic/tables/*.csv`.

  The header of `tables.md` states:
  - the paths completed per set and the error rows;
  - the λ-fixed count and the median clip share;
  - the V6 share of paths with a name whose online β ≤ 0, next to the real run;
  - that random and bandit cells are medians over 5 seeds within a path, then over paths.

  Then, for each set (typical, burst stress, bear stress) and each hedge setting (off, on), these tables:
  1. wealth, median;
  2. wealth, share beating SPY;
  3. growth, max drawdown and Sharpe as bands;
  4. the head-to-heads.

  The real-percentile grid follows, for the typical set only. The §11 caveats close the file.

- [ ] **Step 6: Run it on the smoke rows (Wed), then on the full rows (Thu)**

Run: `julia --project=code scripts/23_synthetic_tables.jl results/synthetic/smoke/rows`, then
`julia --project=code scripts/23_synthetic_tables.jl`
Expected: on the full rows, every table has 14 picker rows plus 2 reference rows and 7 utility columns, and the path
counts match Task 10's count.

- [ ] **Step 7: Run the full suite, then commit**

Run: `julia --project=code code/test/runtests.jl` (expect exit 0)

```bash
git add code/src/SyntheticTables.jl code/src/Include.jl code/test/test_synthetic_tables.jl code/test/runtests.jl scripts/23_synthetic_tables.jl results/synthetic/tables.md results/synthetic/tables
git commit -m "results: synthetic stress-test tables (typical, burst and bear sets)"
```

- [ ] **Step 8: Report to the user (Thu Oct 29).** Use plain words and lead with the answer.
  - Give the hedge-off and hedge-on wealth tables for each set.
  - Give three head-to-head lines: global γ against random, the sector quota, and the hedge.
  - Say where the real run falls among the typical paths.
  - State the §11 caveats.
  - Ask one question: what each slide figure should show. No figures before the answer. No push.
