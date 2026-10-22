# CES Elasticity Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the weight-based CES allocator (`CESw`) beside the share-count CES. Then show both versions' η limits on the L8b list (step (a)), sweep η inside the study with the 10% cap on and off (step (b)), and split the sweep by market state. Finally, rerun every grid once.

**Architecture:**
- `CESw` is a new `AbstractPortfolioRule`. It shares CES's power-cone code through a helper whose cone variable is rescaled to O(1), and closed-form budget-only weights let the tests and the step (a) gate check the solver at every η.
- New per-run statistics come from the existing run files: a scoped solver-outcome log, effective names and market-state growth.
- The two new experiments, `eta` and `limits`, are run lists built from `tier1_runs`, run by `scripts/11_run_study.jl` and summarized by package table functions.

**Tech Stack:** Julia 1.12.7. Package `VLINFORMS2026` in `code/` (JuMP, Clarabel, then SCS as fallback). Figures use Plots.jl in `scripts/`.

**Spec:** `docs/superpowers/specs/2026-10-21-ces-eta-sweep-design.md` (approved 2026-10-22). It extends `2026-10-20-screened-utility-panel-study-design.md` and `2026-10-20-ticker-picker-addendum-design.md`.

## Global Constraints

- **Growth rates only.** Use g = ln(S_end/S_start)/(k·Δt) with Δt = 1/252, in 1/yr. No column, label, caption or name says "return".
- **The data.** The study compares methods on the split-adjusted data as given.
  - Never call split-adjusted prices look-ahead.
  - Never add as-traded prices or a split calendar.
  - Never call the `l8b_13` list hindsight; it was chosen in advance, without 2025 data.
- **Unchanged rules.** No change to share-count CES's theory, γ, the screens, triggers, costs or hedge rules. Rescaling CES's cone is numerical only, and its argmax is unchanged.
- **η values:**
  - Headline η is `ces_eta = 0.5` for both versions in every main table and figure. The sweep is sensitivity around it and never replaces it.
  - η grids: step (b) `[0.1, 0.25, 0.5, 2.0, 4.0, 10.0]`; step (a) `[0.05, 0.1, 0.25, 0.5, 2.0, 4.0, 10.0, 20.0]`. Cobb-Douglas is η = 1.
- **Tolerances.** Never loosen a tolerance silently. An η that fails its closed-form check after rescaling is reported to the user, who decides; it is not dropped automatically.
- **Code style:**
  - Match the code: `My*` mutable types with `new()` constructors, `build(::Type{T}, data::NamedTuple)` factories, units in every docstring.
  - Choices this plan makes are marked **(plan decision)** in docstrings and listed in the README.
- **Turnover settings.** Keep the setting and field names `turnover_frac` and `K_turnover`: they are part of every run's hash and every summary's columns.
- **Never commit** `data/`, `results/runs/` or `results/smoke/`.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The branch is `feat/ces-eta-sweep`.
- **Test commands:**
  - One file: `julia --project=code code/test/<file>.jl`
  - Suite: `julia --project=code code/test/runtests.jl`. The baseline is 1,161 passing tests, exit 0.

## Review Focus

1. **Hash sharing between grids.** If an `eta` or `limits` run shared a hash with a Tier 1, fixed-schedule or universe run, `--eta --fresh` would overwrite that run's file under an η label. The Tier 1 summary would then carry η labels and `15_regression_check.jl` would mis-join. Test: Task 6.
2. **Extreme η with a binding β cap and a free hedge name.** Hedge on is half of step (b). Both versions at η = 0.05 and 20 must solve, meet the cap and leave the non-pool γ ≤ 0 name at ε. Test: Task 2.
3. **Cap off: concentration has to stop at the volatility cap.** CES and CESw at η = 10 with w_max = 1 must land on the cap and not above it. Test: Task 2.
4. **A failed hedge sub-solve under an `:optimal` decision.** `_solve_utility` drops `_min_hedge`'s status, so a failed sub-solve still reports `:optimal`. It must count in `n_fallback`, not in `n_failed`, and must not vanish. Test: Task 4.
5. **The step (a) day rule when the run start has fewer than 5 names at γ > 0.** It moves to the first row with the most. Test: Task 7, as a pure-function test.

---

### Task 1: Closed-form budget weights; CES accurate at every η

**Files:**
- Modify: `code/src/Allocators.jl` (CES `allocate`, new `_ces_objective`, new `budget_only_weights`)
- Modify: `code/src/Include.jl` (export `budget_only_weights`)
- Create: `code/test/test_allocators_eta.jl`; Modify: `code/test/runtests.jl` (include it after `test_allocators_caps.jl`)

**Interfaces:**
- Produces:
  - `budget_only_weights(rule, γ::AbstractVector, S::AbstractVector; W::Real, ε::Real=1e-3)::Vector{Float64}`, with methods for `MyCobbDouglasRule` and `MyCESRule` (Task 2 adds `MyCESWeightRule`). These are dollar weights under the budget alone.
  - `_ces_objective(η::Float64, a::Vector{Float64})::Function`, the `objective!` closure for `_solve_utility` over zₖ = aₖ·xₖ.

- [ ] **Step 1: Write the failing tests** in `code/test/test_allocators_eta.jl`

```julia
using Test, LinearAlgebra, Random, JuMP, VLINFORMS2026
include("helpers.jl")
const ETAS = [0.05, 0.1, 0.25, 0.5, 2.0, 4.0, 10.0, 20.0]          # both spec grids
const CASES = (([0.2, 0.5, 0.3], [10.0, 20.0, 40.0], 1000.0),
    (0.05 .+ 0.2 .* rand(MersenneTwister(5660), 13), 10.0 .+ 490.0 .* rand(MersenneTwister(5661), 13), 1e5))

@testset "budget_only_weights: the L8b closed forms" begin
    γ = [0.2, 0.5, 0.3]; S = [10.0, 20.0, 40.0]
    @test budget_only_weights(MyCobbDouglasRule(), γ, S; W=1000.0) ≈ γ ./ sum(γ)
    for η in ETAS
        n = (γ ./ S) .^ η
        @test budget_only_weights(build(MyCESRule, (η=η,)), γ, S; W=1000.0) ≈ S .* n ./ sum(S .* n)
    end
    @test budget_only_weights(build(MyCESRule, (η=1.0,)), γ, S; W=1000.0) ≈ γ ./ sum(γ)
    # a γ ≤ 0 name holds ε shares, paid first: 5 shares × 20 USD, then 900 USD split 0.4 / 0.6
    @test budget_only_weights(MyCobbDouglasRule(), [0.2, -0.5, 0.3], S; W=1000.0, ε=5.0) ≈ [0.36, 0.1, 0.54]
    # no γ > 0: only the pins
    @test budget_only_weights(MyCobbDouglasRule(), [-0.2, 0.0, -0.3], S; W=1000.0) ≈ 1e-3 .* S ./ 1000
end

@testset "CES matches nᵢ ∝ (γᵢ/Sᵢ)^η at every η (budget only)" begin
    for (γ, S, W) in CASES, η in ETAS
        rule = build(MyCESRule, (η=η,))
        r = allocate(rule, P(γ=γ, prices=S, W=W, σ_g_max=Inf))
        @test r.status == :optimal
        @test r.w ≈ budget_only_weights(rule, γ, S; W=W) atol=1e-4
    end
    for η in (0.05, 20.0)                                             # with a pinned name
        rule = build(MyCESRule, (η=η,)); r = allocate(rule, P(γ=[0.2, -0.5, 0.3], σ_g_max=Inf))
        @test r.n[2] == 1e-3 && r.w ≈ budget_only_weights(rule, [0.2, -0.5, 0.3], [10.0, 20.0, 40.0]; W=1000.0) atol=1e-4
    end
end
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `julia --project=code code/test/test_allocators_eta.jl`
Expected: FAIL with `UndefVarError: budget_only_weights`.

- [ ] **Step 3: Implement `budget_only_weights` in `code/src/Allocators.jl`**

Put it after the Stone–Geary `allocate`. Start from the budget left after the pins: B = W − Σ_{γᵢ≤0} ε·Sᵢ. Names with γᵢ ≤ 0 get the pin weight ε·Sᵢ/W. The names with γᵢ > 0 split B/W as follows:

- Cobb-Douglas: in proportion to γᵢ.
- CES: in proportion to Sᵢ·(γᵢ/Sᵢ)^η, which is nᵢ ∝ (γᵢ/Sᵢ)^η.

With no γ > 0, return the pin weights only, matching `allocate`'s `:no_preferred`. Add `budget_only_weights` to the Allocators export line in `Include.jl`.

- [ ] **Step 4: Run again; the closed-form testset passes and CES fails at the extreme η**

Run: `julia --project=code code/test/test_allocators_eta.jl`

Expected: the first testset passes, and the CES testset fails at least at η = 0.05. The current scale c = W/mean(S) leaves n/c ≈ 1/K at the optimum, so the cone holds (n/c)^(−19), about 1e21 for 13 names. Write down which η fail; they go in the commit message. If none fail, still do Step 5, because CESw needs the shared helper, and say so in the commit message.

- [ ] **Step 5: Implement `_ces_objective(η, a)` and route CES through it**

Both cones work on the scaled variable zₖ = aₖ·xₖ:
- η > 1: add `[a[k]*x[k], 1.0, t[k]] in MOI.PowerCone(ρ)` and return Σγₖtₖ.
- η < 1: add `[t[k], a[k]*x[k], 1.0] in MOI.PowerCone(η)`, so t ≥ z^ρ, and return −Σγₖtₖ.

CES passes `a = fill(sum(S_pref) / W, Kp)`, so z = 1 at the book with equal share counts over the preferred names.

**(plan decision)** The scale c = W/Σ_{γ>0}Sᵢ replaces W/mean(S). Rewrite the CES docstring's scale sentence to say this, and say that the η > 1 branch is now scaled too.

- [ ] **Step 6: Run the new file and every allocator test**

Run:
```
julia --project=code code/test/test_allocators_eta.jl
julia --project=code code/test/test_allocators_utility.jl
julia --project=code code/test/test_allocators_hedge.jl
julia --project=code code/test/test_allocators_caps.jl
```

Expected: all pass. This includes "CES η = 0.5 at W = 1e6 (budget only)" at rtol 1e-6.

If an η still fails after rescaling, stop. Report the η, the version and the largest weight error to the user, and do not loosen `atol` (spec §10).

- [ ] **Step 7: Commit**

```bash
git add code/src/Allocators.jl code/src/Include.jl code/test/test_allocators_eta.jl code/test/runtests.jl
git commit -m "feat(allocators): closed-form budget weights; CES cone scaled for every η"
```

---

### Task 2: The weight-based CES allocator

**Files:**
- Modify: `code/src/Allocators.jl` (type, `build`, `allocate`, `budget_only_weights` method)
- Modify: `code/src/Include.jl` (export `MyCESWeightRule`)
- Test: `code/test/test_allocators_eta.jl`; Modify: `code/test/test_allocators_hedge.jl` (`RULES_LONG`)

**Interfaces:**
- Consumes: `_ces_objective`, `budget_only_weights`, `_solve_utility` (Task 1).
- Produces: `MyCESWeightRule` with field `η::Float64`, built from `(η = …,)` (finite, > 0), with `allocate(rule::MyCESWeightRule, p)` and `budget_only_weights(rule::MyCESWeightRule, γ, S; W, ε)`.

- [ ] **Step 1: Write the failing tests** (append to `test_allocators_eta.jl`)

```julia
@testset "CESw matches wᵢ ∝ γᵢ^η at every η (budget only)" begin
    @test budget_only_weights(build(MyCESWeightRule, (η=2.0,)), [0.2, 0.5, 0.3], [10.0, 20.0, 40.0]; W=1000.0) ≈
        [0.04, 0.25, 0.09] ./ 0.38
    for (γ, S, W) in CASES, η in ETAS
        rule = build(MyCESWeightRule, (η=η,))
        r = allocate(rule, P(γ=γ, prices=S, W=W, σ_g_max=Inf))
        @test r.status == :optimal
        @test r.w ≈ γ .^ η ./ sum(γ .^ η) atol=1e-4
    end
    for η in (0.05, 20.0)
        rule = build(MyCESWeightRule, (η=η,)); r = allocate(rule, P(γ=[0.2, -0.5, 0.3], σ_g_max=Inf))
        @test r.n[2] == 1e-3 && r.w ≈ budget_only_weights(rule, [0.2, -0.5, 0.3], [10.0, 20.0, 40.0]; W=1000.0) atol=1e-4
    end
    @test allocate(build(MyCESWeightRule, (η=1.0,)), P()).n ≈ allocate(MyCobbDouglasRule(), P()).n rtol=1e-8
    @test_throws ArgumentError build(MyCESWeightRule, (η=0.0,))
end

@testset "a split changes CES's weights, not CESw's" begin
    S2 = [1.0, 20.0, 40.0]                                            # name 1 splits 10:1, γ unchanged
    for η in (0.25, 4.0)
        cw = build(MyCESWeightRule, (η=η,)); ces = build(MyCESRule, (η=η,))
        @test allocate(cw, P(prices=S2, σ_g_max=Inf)).w ≈ allocate(cw, P(σ_g_max=Inf)).w atol=1e-5
        @test maximum(abs.(allocate(ces, P(prices=S2, σ_g_max=Inf)).w .- allocate(ces, P(σ_g_max=Inf)).w)) > 0.05
    end
end

@testset "CESw under the shared limits" begin
    Σ = Diagonal([9.0, 16.0, 25.0])
    for η in (0.5, 2.0)
        cw = build(MyCESWeightRule, (η=η,))
        @test maximum(allocate(cw, P(w_max=0.4)).w) ≤ 0.4 + 1e-6
        r = allocate(cw, P(Σ_g=Σ, σ_g_max=1.5)); @test sqrt(r.w' * Σ * r.w) ≈ 1.5 rtol=1e-5
        @test sum(allocate(cw, P(sector=["A", "A", "B"], s_max=0.5)).w[1:2]) ≈ 0.5 rtol=1e-5
        @test allocate(cw, P(γ=[0.2, -0.5, 0.3])).n[2] == 1e-3
        r = allocate(cw, P(γ=[-0.1, 0.0, -0.3])); @test r.status == :no_preferred && all(r.n .== 1e-3)
    end
end

@testset "extreme η with a binding β cap and a hedge name" begin              # Review Focus 2
    γ = [0.5, 0.4, -0.2, -0.1]; β = [1.2, 1.0, -0.5, -0.5]; S = [10.0, 20.0, 40.0, 25.0]
    for R in (MyCESRule, MyCESWeightRule), η in (0.05, 20.0)
        r = allocate(build(R, (η=η,)), P(γ=γ, prices=S, β=β, β_max=0.4, hedge=[false, false, true, false], w_max=0.5))
        @test r.status == :optimal && dot(r.w, β) ≤ 0.4 + 1e-6 && r.n[4] == 1e-3
    end
end

@testset "cap off: concentration stops at the volatility cap" begin            # Review Focus 3
    Σ = Diagonal([9.0, 16.0, 25.0])
    for R in (MyCESRule, MyCESWeightRule)
        r = allocate(build(R, (η=10.0,)), P(Σ_g=Σ, σ_g_max=1.5, w_max=1.0))
        @test r.status == :optimal && sqrt(r.w' * Σ * r.w) ≤ 1.5 + 1e-6 && sqrt(r.w' * Σ * r.w) ≈ 1.5 rtol=1e-5
    end
end
```

In `test_allocators_hedge.jl`, append `build(MyCESWeightRule, (η=0.5,)), build(MyCESWeightRule, (η=2.0,))` to `RULES_LONG`. That runs CESw through the existing β-cap, hedge-floor and W = 1e5 least-hedge tests.

- [ ] **Step 2: Run the tests and watch them fail**

Run: `julia --project=code code/test/test_allocators_eta.jl`
Expected: FAIL with `UndefVarError: MyCESWeightRule`.

- [ ] **Step 3: Implement `MyCESWeightRule` in `code/src/Allocators.jl`**

The docstring goes beside `MyCESRule`. It says: U = (Σ_{γᵢ>0} γᵢ·wᵢ^ρ)^(1/ρ), with ρ = 1 − 1/η and wᵢ = nᵢSᵢ/W; under a budget alone wᵢ ∝ γᵢ^η; every other rule is CES's.

`build` checks the same things as `MyCESRule`'s. `allocate` does the following:
- At η = 1, return `allocate(MyCobbDouglasRule(), p)`.
- Pin names at ε; return `:no_preferred` when no γ > 0.
- Otherwise call `_solve_utility(p, pins, zeros(Kp), _ces_objective(η, Kp .* p.prices[pref] ./ p.W))`. This makes z = Kp·wᵢ, which is 1 at equal weights **(plan decision)**.

`budget_only_weights` for this rule splits B/W in proportion to γᵢ^η. Export `MyCESWeightRule` beside `MyCESRule`.

- [ ] **Step 4: Run the allocator tests**

Run: `julia --project=code code/test/test_allocators_eta.jl` and `julia --project=code code/test/test_allocators_hedge.jl`
Expected: all pass. If an η fails, follow Task 1, Step 6: report it and do not loosen the tolerance.

- [ ] **Step 5: Commit**

```bash
git add code/src/Allocators.jl code/src/Include.jl code/test/test_allocators_eta.jl code/test/test_allocators_hedge.jl
git commit -m "feat(allocators): weight-based CES (CESw), price-invariant, with every CES rule"
```

---

### Task 3: CESw in the study grids

**Files:**
- Modify: `code/src/StudyBacktest.jl` (`_STUDY_ALLOCATORS`, `make_rule`, `MyStudyRun` and `build` docstrings)
- Modify: `config/study.toml` (`[grid] allocators`)
- Modify: `code/src/StudyTables.jl` (`TABLE_ALLOCATOR_ORDER`)
- Test: `code/test/test_study_backtest.jl`, `code/test/test_study_config.jl`, `code/test/test_study_tables.jl`

**Interfaces:**
- Consumes: `MyCESWeightRule` (Task 2).
- Produces:
  - `make_rule("CESw", s)` returns `build(MyCESWeightRule, (η = s.ces_eta,))`.
  - The allocator order everywhere is `["CD", "CES", "CESw", "SG", "MinVar", "Tangent", "LS"]`.

- [ ] **Step 1: Write the failing tests**

```julia
# test_study_backtest.jl
@testset "CESw in the study" begin
    @test make_rule("CESw", with_settings(MyStudySettings(); ces_eta=2.0)) isa MyCESWeightRule
    @test make_rule("CESw", with_settings(MyStudySettings(); ces_eta=2.0)).η == 2.0
    md, ctx = study_ctx(); res = run_study(ctx, study_run(ctx, "CESw", :gamma))
    @test res.decisions.reason[1] == :initial && minimum(res.cash) ≥ -1e-6
    @test res.equity ≈ res.cash .+ held_value(res, md) rtol=1e-9
end
# test_study_config.jl, in "settings and config file": replace `length(cfg.grid.allocators) == 6` with
@test cfg.grid.allocators == ["CD", "CES", "CESw", "SG", "MinVar", "Tangent", "LS"]
# test_study_tables.jl, in "tables are sorted deterministically"
@test sort_table(DataFrame(allocator=["LS", "CESw", "CES", "CD"])).allocator == ["CD", "CES", "CESw", "LS"]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `julia --project=code code/test/test_study_backtest.jl` (and the config and tables files)
Expected: FAIL. `make_rule` throws `ArgumentError: unknown allocator "CESw"`, and the config and order checks fail.

- [ ] **Step 3: Implement**

- Add `"CESw"` after `"CES"` in `_STUDY_ALLOCATORS`, `TABLE_ALLOCATOR_ORDER` and `config/study.toml`'s `allocators`.
- Add the `make_rule` branch.
- Update the allocator lists in the docstrings of `MyStudyRun`, its `build` and `make_rule`.

- [ ] **Step 4: Run the three files; they pass**

- [ ] **Step 5: Commit**

```bash
git add code/src/StudyBacktest.jl code/src/StudyTables.jl config/study.toml code/test/test_study_backtest.jl code/test/test_study_config.jl code/test/test_study_tables.jl
git commit -m "feat(study): CESw joins every grid at the headline η"
```

---

### Task 4: Solver outcomes and concentration in the run statistics

**Files:**
- Modify: `code/src/Allocators.jl` (`_SOLVER_LOG`, `_solver_outcome`, `_worst_solver`; `_solve_with_fallback` records)
- Modify: `code/src/StudyBacktest.jl` (`_study_decision` collects the log; new decisions column `solver`; `MyStudyResult` and `_study_decision` docstrings)
- Modify: `code/src/StudyRunner.jl` (`_eff_names`, `study_stats`, `_STUDY_STAT_FIELDS`)
- Modify: `code/test/helpers.jl` (`hand_result`)
- Test: `code/test/test_allocators_eta.jl`, `code/test/test_study_runner.jl`, `code/test/test_study_backtest.jl`

**Interfaces:**
- Produces:
  - **`_SOLVER_LOG`:** `const _SOLVER_LOG = Base.ScopedValues.ScopedValue{Union{Nothing,Vector{Symbol}}}(nothing)`. Inside `Base.ScopedValues.with(_SOLVER_LOG => log) do … end`, every `_solve_with_fallback` call pushes one outcome to `log`; outside such a scope nothing is recorded. Use the qualified names: the test helpers define their own `with`.
  - **`_solver_outcome`:** `_solver_outcome(statuses::AbstractVector{MOI.TerminationStatusCode})::Symbol` maps the statuses of the solvers tried, in order, to an outcome:

    | Statuses | Outcome |
    |---|---|
    | `[OPTIMAL]` | `:clarabel` |
    | `[ALMOST_OPTIMAL]` | `:clarabel_almost` |
    | Clarabel not accepted, SCS accepted | `:scs` |
    | neither accepted | `:failed` |

  - **`_worst_solver`:** `_worst_solver(outcomes::AbstractVector{Symbol})::Symbol` returns the worst outcome in the order `(:none, :clarabel, :clarabel_almost, :scs, :failed)`, or `:none` when the list is empty.
  - **Decisions column:** `res.decisions.solver::Vector{Symbol}` holds the worst outcome of each decision's solves. It is `:none` when nothing was solved (pins, or an empty 130/30).
  - **`study_stats` fields:**
    - `mean_eff_names` (names): the mean over the counted decisions of 1/Σw̃ᵢ², where w̃ are the target weights above `_BT_HELD_W` rescaled to sum to 1. A decision whose long weights above `_BT_HELD_W` sum to less than 0.01 is skipped.
    - `n_eff_skipped` (count of skipped decisions).
    - `n_failed`: decisions with status `:infeasible` or `:solver_failed`.
    - `n_fallback`: decisions not failed whose solver is `:clarabel_almost`, `:scs` or `:failed`, the last being a failed sub-solve.
    - `n_almost_optimal`: decisions not failed whose solver is `:clarabel_almost`.
  - **`_eff_names`:** `_eff_names(w::AbstractVector{<:Real})::Float64` returns that per-decision value, or NaN when the decision is skipped.
  - **Test helper:** `hand_result(; dates::Vector{Date}, equity=fill(1e5, length(dates)), xi=zeros(length(dates)), decisions::DataFrame, targets=Dict{Date,Dict{String,Float64}}())::MyStudyResult` in `helpers.jl`. Every other field is empty or zero, and the repicks hold one `:initial` row with basket `["A"]`.

- [ ] **Step 1: Write the failing tests**

```julia
# test_allocators_eta.jl
@testset "solver outcomes" begin
    V = VLINFORMS2026
    @test V._solver_outcome([MOI.OPTIMAL]) == :clarabel
    @test V._solver_outcome([MOI.ALMOST_OPTIMAL]) == :clarabel_almost
    @test V._solver_outcome([MOI.SLOW_PROGRESS, MOI.OPTIMAL]) == :scs
    @test V._solver_outcome([MOI.SLOW_PROGRESS, MOI.ALMOST_OPTIMAL]) == :scs
    @test V._solver_outcome([MOI.INFEASIBLE, MOI.INFEASIBLE]) == :failed
    @test V._worst_solver(Symbol[]) == :none && V._worst_solver([:clarabel, :scs, :clarabel_almost]) == :scs
    log = Symbol[]
    r = Base.ScopedValues.with(() -> allocate(MyCobbDouglasRule(), P(σ_g_max=Inf)), V._SOLVER_LOG => log)
    @test r.status == :optimal && length(log) == 1 && log[1] in (:clarabel, :clarabel_almost)
    allocate(MyCobbDouglasRule(), P(σ_g_max=Inf))                     # outside a scope: not recorded
    @test length(log) == 1
end

# test_study_runner.jl
@testset "concentration and solver counts (hand-built result)" begin
    d = Date(2025, 1, 2) .+ Day.(0:3)
    dec = DataFrame(date=d[1:3], reason=[:initial, :band_exit, :cash_revisit],
        status=[:optimal, :solver_failed, :optimal], n_names=[2, 1, 0], invested_frac=[0.9, 0.5, 0.005],
        μ_T=zeros(3), σ_T=zeros(3), beta_target=zeros(3), beta_max=fill(Inf, 3), cap_binds=falses(3),
        hedge_weight=zeros(3), solver=[:clarabel_almost, :failed, :failed])
    targets = Dict(d[1] => Dict("A" => 0.5, "B" => 0.5, "C" => 0.0005, "D" => -0.1),   # C below 0.1%, D short
        d[2] => Dict("A" => 0.5), d[3] => Dict("A" => 0.005))                           # d[3]: long 0.005 < 1%
    st = study_stats(hand_result(dates=collect(d), decisions=dec, targets=targets))
    @test st.mean_eff_names ≈ 1.5 && st.n_eff_skipped == 1            # decisions 1 and 2: 2 names and 1
    @test st.n_failed == 1                                             # the :solver_failed decision
    @test st.n_fallback == 2      # decision 1 (ALMOST_OPTIMAL) and decision 3 (failed hedge sub-solve, Review Focus 4)
    @test st.n_almost_optimal == 1
end

# test_study_backtest.jl, in "CESw in the study"
@test all(in((:none, :clarabel, :clarabel_almost, :scs, :failed)), res.decisions.solver)
@test all(res.decisions.solver[res.decisions.status .== :optimal] .!= :none)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `julia --project=code code/test/test_allocators_eta.jl`
Expected: FAIL with `UndefVarError: _solver_outcome`. The runner file fails on `hand_result`, then on the missing stats fields.

- [ ] **Step 3: Implement**

- **Solver log.** In `_solve_with_fallback`, record `_solver_outcome(statuses)` into `_SOLVER_LOG[]` (when it is not `nothing`) just before each return. Its return value and the statuses it accepts are unchanged.
- **Decisions.** `_study_decision` wraps `allocate` in a fresh log and returns `solver = _worst_solver(log)`. The empty-columns path returns `:none`. `run_study` pushes the value into a new `dec.solver` column.
- **Stats.** `study_stats` computes the new fields from `res.targets[d]` for each `d` in `dec.date`, and from `dec.status` and `dec.solver`. Append `:mean_eff_names, :n_eff_skipped, :n_failed, :n_fallback, :n_almost_optimal` to `_STUDY_STAT_FIELDS`.
- **Docstrings.** Document the new column and fields, with units. Mark the definitions of `n_fallback` and `mean_eff_names` **(plan decision)**.

- [ ] **Step 4: Run the three files; they pass**

- [ ] **Step 5: Commit**

```bash
git add code/src/Allocators.jl code/src/StudyBacktest.jl code/src/StudyRunner.jl code/test/helpers.jl code/test/test_allocators_eta.jl code/test/test_study_runner.jl code/test/test_study_backtest.jl
git commit -m "feat(study): record solver outcomes; effective names and failure counts per run"
```

---

### Task 5: Market-state growth per run

**Files:**
- Modify: `code/src/StudyRunner.jl` (`regime_stats`, `run_and_save_study`, `_STUDY_STAT_FIELDS`)
- Modify: `code/src/Include.jl` (export `regime_stats`)
- Test: `code/test/test_study_runner.jl`

**Interfaces:**
- Consumes: `hand_result` (Task 4).
- Produces: `regime_stats(res::MyStudyResult, spy_close::AbstractVector{<:Real})::NamedTuple{(:g_bull, :g_bear, :n_bull, :n_bear, :spy_g_bull, :spy_g_bear)}`.
  - `spy_close` has one entry per entry of `res.dates`.
  - Each day k ≥ 2 has growth ln(V_k/V_{k−1})·252 (yr⁻¹). The day is bullish when `res.daily.xi[k−1] < 0`, bearish when it is > 0, and dropped when it is 0.
  - `g_*` and `spy_g_*` are the means over each label's days, NaN when there are none; `n_*` are day counts.
  - `run_and_save_study` stores `merge(study_stats(res), regime_stats(res, md.spy_close[i0:i1]))` as `stats`.

- [ ] **Step 1: Write the failing test**

```julia
@testset "market-state growth: day t labeled by ξ on day t−1" begin
    d = collect(Date(2025, 1, 2) .+ Day.(0:4))
    res = hand_result(dates=d, equity=[100.0, 101.0, 100.0, 102.0, 102.0], xi=[-0.5, 0.3, 0.0, -0.2, 0.9],
        decisions=DataFrame(date=Date[], reason=Symbol[], status=Symbol[], n_names=Int[], invested_frac=Float64[],
            μ_T=Float64[], σ_T=Float64[], beta_target=Float64[], beta_max=Float64[], cap_binds=Bool[],
            hedge_weight=Float64[], solver=Symbol[]))
    spy = [10.0, 10.0, 11.0, 11.0, 10.0]
    g(a, b) = log(b / a) * 252
    r = regime_stats(res, spy)                    # days 2, 5 bullish; day 3 bearish; day 4 dropped (ξ = 0)
    @test r.n_bull == 2 && r.n_bear == 1
    @test r.g_bull ≈ (g(100, 101) + g(102, 102)) / 2 && r.g_bear ≈ g(101, 100)
    @test r.spy_g_bull ≈ (g(10, 10) + g(11, 10)) / 2 && r.spy_g_bear ≈ g(10, 11)
    res.daily.xi .= -1.0
    @test isnan(regime_stats(res, spy).g_bear) && regime_stats(res, spy).n_bear == 0
end
```

In the existing "study runner" testset, after `df = rebuild_study_summary(out, "t")`, add:
`@test all(c -> c in names(df), ["mean_eff_names", "n_failed", "n_fallback", "n_almost_optimal", "g_bull", "spy_g_bear"])`.

- [ ] **Step 2: Run them and watch them fail** (`UndefVarError: regime_stats`)

Run: `julia --project=code code/test/test_study_runner.jl`

- [ ] **Step 3: Implement**

Add `regime_stats`, merge it in `run_and_save_study`, and append `:g_bull, :g_bear, :n_bull, :n_bear, :spy_g_bull, :spy_g_bear` to `_STUDY_STAT_FIELDS`. The docstring cites spec §7 and the ξ_{t−1} rule **(plan decision)**.

- [ ] **Step 4: Run; it passes**

- [ ] **Step 5: Commit**

```bash
git add code/src/StudyRunner.jl code/src/Include.jl code/test/test_study_runner.jl
git commit -m "feat(study): growth by market state, each day labeled by the previous day's ξ"
```

---

### Task 6: η grids, run lists, the command line, and the turnover relabel

**Files:**
- Modify: `config/study.toml`. Add three `[grid]` keys, placed *before* the `[grid.fixed_lists]` header (TOML would otherwise put them in the subtable), and the turnover comment.
- Modify: `code/src/StudySettings.jl` (`load_study` and its docstring; the `MyStudySettings` docstring for the relabel)
- Modify: `code/src/StudyRunner.jl` (`_eta_text`, `eta_runs`, `limits_runs`)
- Modify: `code/src/Include.jl` (export `eta_runs`, `limits_runs`)
- Modify: `scripts/11_run_study.jl` (the `--eta` and `--limits` flags and the header comment)
- Modify: `code/src/Allocators.jl` and `code/src/StudyBacktest.jl` (relabel text only)
- Test: `code/test/test_study_config.jl`, `code/test/test_study_runner.jl`

**Interfaces:**
- Consumes: `tier1_runs`, `_study_label`, `with_settings`.
- Produces:
  - **Config keys:**
    ```toml
    eta = [0.1, 0.25, 0.5, 2.0, 4.0, 10.0]                     # CES elasticities of the eta sweep, step (b) (dimensionless); Cobb-Douglas is eta = 1
    eta_limits = [0.05, 0.1, 0.25, 0.5, 2.0, 4.0, 10.0, 20.0]  # CES elasticities of the clean-limits runs, step (a) (dimensionless)
    limits_list = "l8b_13"                                      # fixed list of the clean-limits runs (a [grid.fixed_lists] name)
    ```
  - **`load_study`:** `grid` gains `eta::Vector{Float64}`, `eta_limits::Vector{Float64}` and `limits_list::String`. It throws `ArgumentError` when a key is missing, an η is not finite and > 0, or `limits_list` names no fixed list.
  - **`eta_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}`**, 2,806 runs with the default grid:
    - Cap on: `"CES"` and `"CESw"` at every `grid.eta` except `s.ces_eta`, on `with_settings(s; ces_eta=η)`.
    - Cap off: `"CD"` once, plus `"CES"` and `"CESw"` at every `grid.eta`, on `with_settings(s; ces_eta=η, w_max=1.0)`.
    - Each is `tier1_runs` with `allocators=[a]`, relabeled `<tier1 label>|eta=<η>|cap=<on|off>`. CD's η is 1.
  - **`limits_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}`**, 17 runs:
    - `"CD"` (η = 1), plus `"CES"` and `"CESw"` at every `grid.eta_limits`.
    - Each runs on screen `:fixed` with seed `grid.base_seed`, on `with_settings(s; fixed_tickers=list, basket_K=length(list), hedge=false, w_max=1.0, vol_cap_annual=Inf, s_max=1.0, ces_eta=η)`, where `list` is `grid.limits_list`'s tickers.
    - Labels look like `<a>|fixed:l8b_13|hedge=false|seed=5660|eta=<η>|limits`.
  - **`_eta_text(η::Real)::String`** writes whole numbers without a decimal point: `2.0 → "2"`, `0.05 → "0.05"`.
  - **The script:** `scripts/11_run_study.jl --eta` writes experiment `eta` (`results/eta_summary.csv`), and `--limits` writes experiment `limits` (`results/limits_summary.csv`). At most one of `--fixed`, `--universe`, `--smoke`, `--eta` and `--limits` may be passed.

- [ ] **Step 1: Write the failing tests**

```julia
# test_study_config.jl, in "addendum settings and grid"
@test cfg.grid.eta == [0.1, 0.25, 0.5, 2.0, 4.0, 10.0]
@test cfg.grid.eta_limits == [0.05, 0.1, 0.25, 0.5, 2.0, 4.0, 10.0, 20.0] && cfg.grid.limits_list == "l8b_13"
src = read(joinpath(REPO, "config", "study.toml"), String)
for (from, to) in ("limits_list = \"l8b_13\"" => "limits_list = \"nope\"", "eta = [0.1," => "eta = [-0.1,")
    p = joinpath(mktempdir(), "s.toml"); write(p, replace(src, from => to))
    @test_throws ArgumentError load_study(p)
end

# test_study_runner.jl
@testset "η sweep and clean-limits run lists" begin
    s = MyStudySettings()
    g = (n_seeds=30, base_seed=5660, allocators=["CD", "CES", "CESw", "SG", "MinVar", "Tangent", "LS"],
        screens=[:bandit, :gamma, :random], hedge=[true, false], K=[5, 8, 13, 21, 34],
        universe_screens=[:global_gamma, :liquid_gamma, :corr_gamma, :random_global, :random_liquid],
        fixed_lists=["l8b_13" => s.fixed_tickers, "l8b_15" => vcat(s.fixed_tickers, ["WMT", "PG"])],
        eta=[0.1, 0.25, 0.5, 2.0, 4.0, 10.0], eta_limits=[0.05, 0.1, 0.25, 0.5, 2.0, 4.0, 10.0, 20.0],
        limits_list="l8b_13")
    e = eta_runs(s, g)
    on = filter(r -> r.settings.w_max == s.w_max, e); off = filter(r -> r.settings.w_max == 1.0, e)
    @test length(e) == 2806 && length(on) == 1220 && length(off) == 1586
    @test all(r -> r.allocator in ("CES", "CESw") && r.settings.ces_eta != 0.5, on)
    @test count(r -> r.allocator == "CD", off) == 122
    labels = Set(r.label for r in e)
    @test "CESw|bandit|hedge=true|seed=5660|eta=2|cap=off" in labels
    @test "CES|gamma|hedge=false|seed=5660|eta=0.25|cap=on" in labels
    @test "CD|random|hedge=true|seed=5689|eta=1|cap=off" in labels
    l = limits_runs(s, g)
    @test length(l) == 17
    @test all(r -> r.screen == :fixed && !r.settings.hedge && r.settings.w_max == 1.0 && r.settings.s_max == 1.0 &&
        r.settings.vol_cap_annual == Inf && r.settings.fixed_tickers == s.fixed_tickers && r.settings.basket_K == 13, l)
    @test "CES|fixed:l8b_13|hedge=false|seed=5660|eta=0.05|limits" in Set(r.label for r in l)
    @test "CD|fixed:l8b_13|hedge=false|seed=5660|eta=1|limits" in Set(r.label for r in l)
    # Review Focus 1: no η or limits run shares a hash with a run of another grid
    others = vcat(tier1_runs(s, g), fixed_schedule_runs(s, g), universe_runs(s, g), fixed_list_runs(s, g))
    h = study_hash.(vcat(e, l))
    @test allunique(h) && isdisjoint(Set(h), Set(study_hash.(others)))
end
```

- [ ] **Step 2: Run them and watch them fail** (`UndefVarError: eta_runs`; the config test fails on a missing field `eta`)

- [ ] **Step 3: Implement**

- Add the config keys, the `load_study` checks, the run lists and the exports.
- In `11_run_study.jl`, add the two flags to the `bad` filter and the `count(...) <= 1` check, and add two branches that set `runs` and `experiment`. Task 7 adds the limits-day step.

- [ ] **Step 4: Relabel the turnover rule** (text only; the user approved it 2026-10-22)

Use these exact phrasings:

- **`config/study.toml`:** `turnover_frac = 0.10      # spread-cost budget per decision: half-spread cost of the share changes <= this x wealth (fraction of wealth)`
- **`MyStudySettings` docstring:** add "`turnover_frac` sets each decision's spread-cost budget, c̄·Σ|Δnᵢ| ≤ `turnover_frac`·W with c̄ the half-spread cost per share at the basket's mean price. It is kept from the trial; at 5 bps, for a long-only book, it can bind only when a basket's mean share price is more than 100 times its cheapest name's."
- **`MyConstraintSet` docstring:** "- `K_turnover`: spread-cost budget (USD), c̄·Σ|nᵢ − n_prevᵢ| ≤ K_turnover"
- **`add_shared_constraints!` docstring:** the bullet "**Turnover budget:**" becomes "**Spread-cost budget** (the trial's turnover rule):"
- **The code comment** "# Turnover budget (ℓ₁ slack …" becomes "# Spread-cost budget (the trial's turnover rule; ℓ₁ slack …"
- **`_study_problem` docstring:** "K_turnover = `turnover_frac`·V (USD)" becomes "spread-cost budget K_turnover = `turnover_frac`·V (USD)".

- [ ] **Step 5: Run the config and runner files; they pass. Then check the script's argument guard**

Run: `julia scripts/11_run_study.jl --eta --fixed`
Expected: `ERROR: pass at most one of --fixed, --universe, --smoke, --eta, --limits`, before any data loads.

- [ ] **Step 6: Commit**

```bash
git add config/study.toml code/src/StudySettings.jl code/src/StudyRunner.jl code/src/Include.jl code/src/Allocators.jl code/src/StudyBacktest.jl scripts/11_run_study.jl code/test/test_study_config.jl code/test/test_study_runner.jl
git commit -m "feat(study): eta and limits run lists and flags; relabel the turnover rule a spread-cost budget"
```

---

### Task 7: The dial on one day (the step (a) picture and its closed-form gate)

**Files:**
- Create: `code/src/Limits.jl`. Include it after `StudyRunner.jl` in `Include.jl` and export `limits_row` and `limits_day`.
- Modify: `scripts/11_run_study.jl` (the `--limits` branch writes `results/limits_day.csv` and applies the gate; add `using CSV`)
- Create: `code/test/test_limits.jl`; Modify: `code/test/runtests.jl`

**Interfaces:**
- Consumes: `_study_problem`, `make_rule`, `budget_only_weights`, `context_key`.
- Produces:
  - **`limits_row(γ::AbstractMatrix{<:Real}, cols::AbstractVector{Int}; min_names::Int=5)::Int`** returns the run row s of the picture. It is 1 when at least `min_names` names in `cols` have γ > 0 on row 1. Otherwise it is the first row with the most names at γ > 0 **(plan decision)**.
  - **`limits_day(ctx::MyStudyContext, runs::Vector{MyStudyRun}, s::Int)::DataFrame`** has one row per run and ticker, with columns `label, allocator, eta, date, ticker, gamma, price, weight, closed_form, status`.
    - Problem: `_study_problem(ctx, run, ctx.rows[s], cols, fill(false, K), zeros(length(ctx.md.tickers)), run.settings.W0; basket=cols)`, where `cols` are the sorted columns of `run.settings.fixed_tickers`. This is a fresh book on day s **(plan decision)**.
    - `weight` is the solver's wᵢ; `closed_form` is `budget_only_weights(rule, p.γ, p.prices; W=p.W, ε=p.constraints.ε)`; `eta` is 1.0 for CD.
    - Throws `ArgumentError` for an allocator outside CD, CES and CESw, a screen other than `:fixed`, or settings whose `context_key` differs from `ctx.settings`'s.

- [ ] **Step 1: Write the failing tests** in `code/test/test_limits.jl`

```julia
using Test, Dates, DataFrames, VLINFORMS2026
include("helpers.jl")

@testset "limits day: the row rule" begin                                       # Review Focus 5
    γ = [0.1 -0.1 -0.1 -0.1 -0.1 -0.1;    # row 1: one name > 0
         0.1  0.1  0.1 -0.1 -0.1 -0.1;    # row 2: three
         0.1  0.1  0.1  0.1 -0.1 -0.1;    # row 3: four, the most
         0.1  0.1  0.1  0.1 -0.1 -0.1]    # row 4: four again, later
    @test limits_row(γ, 1:6) == 3
    @test limits_row(γ, 1:6; min_names=1) == 1
    @test limits_row(γ, [5, 6]) == 1                                  # every row ties at 0: the first
end

@testset "limits day: solver weights match the closed forms and the run's first decision" begin
    md, ctx = study_ctx(; hedge=false, w_max=1.0, vol_cap_annual=Inf, fixed_tickers=md_tickers_13(), basket_K=13)
    runs = [build(MyStudyRun, (settings=with_settings(ctx.settings; ces_eta=η), allocator=a, screen=:fixed,
        seed=5660, label="$(a)|$(η)")) for (a, η) in (("CD", 0.5), ("CES", 0.05), ("CES", 20.0), ("CESw", 0.05),
        ("CESw", 20.0), ("CES", 0.5))]
    day = limits_day(ctx, runs, 1)
    @test nrow(day) == 13 * length(runs) && all(day.date .== md.dates[first(ctx.rows)])
    @test maximum(abs.(day.weight .- day.closed_form)) ≤ 1e-4                     # the real-data gate is 1e-3
    @test all(day.eta[day.allocator .== "CD"] .== 1.0)
    r = runs[end]; res = run_study(ctx, r); w = day[day.label .== r.label, :]
    @test all(isapprox(w.weight[k], res.targets[res.dates[1]][w.ticker[k]]; atol=1e-12) for k in 1:nrow(w))
    @test_throws ArgumentError limits_day(ctx, [study_run(ctx, "SG", :fixed)], 1)
end
```

- [ ] **Step 2: Run it and watch it fail** (`UndefVarError: limits_row`)

Run: `julia --project=code code/test/test_limits.jl`

- [ ] **Step 3: Implement `Limits.jl`; then add the gate to the `--limits` branch of `11_run_study.jl`**

After the grid and summary, the `--limits` branch does the following:
1. Build `ctx = build_context(md, calib, runs[1].settings)`, then `cols` and `s = limits_row(ctx.γ, cols)`.
2. Write `day = limits_day(ctx, runs, s)` to `results/limits_day.csv`.
3. Print the date. When `s > 1`, also print "(fewer than 5 of the 13 names had γ > 0 on the run start)".
4. Call `error(...)`, listing label, ticker, weight and closed form, for every row where `abs(weight − closed_form) > 1e-3` (spec §5).

- [ ] **Step 4: Run; it passes**

- [ ] **Step 5: Commit**

```bash
git add code/src/Limits.jl code/src/Include.jl scripts/11_run_study.jl code/test/test_limits.jl code/test/runtests.jl
git commit -m "feat(limits): one-day dial of both CES versions, gated on the closed forms"
```

---

### Task 8: The sweep tables

**Files:**
- Modify: `code/src/StudyTables.jl` (`eta_frame`, `ETA_METRICS`, `REGIME_METRICS`, `eta_solver_table`, `sort_table` keys)
- Modify: `code/src/Include.jl` (exports)
- Modify: `scripts/12_tables.jl`
- Test: `code/test/test_study_tables.jl`

**Interfaces:**
- Consumes: the summary columns `ces_eta` and `w_max` (settings), the Task 4 and 5 stats, `group_summary`, `load_ok` (script).
- Produces:
  - **`eta_frame(df::DataFrame)::DataFrame`**: the `"ok"` rows (when there is a `status` column) of allocators CD, CES and CESw. It adds `eta::Float64` (1.0 for CD, otherwise `ces_eta`) and `cap::String` (`"on"` when `w_max < 1`, otherwise `"off"`).
  - **`const ETA_METRICS`** `= [:realized_growth, :sharpe, :max_drawdown, :vol, :mean_eff_names, :mean_invested, :turnover]`
  - **`const REGIME_METRICS`** `= [:g_bull, :g_bear, :n_bull, :n_bear, :spy_g_bull, :spy_g_bear]`
  - **`eta_solver_table(df::DataFrame; threshold::Real=0.01)::DataFrame`**: per `allocator, cap, eta` of `eta_frame(df)`, it gives `n_runs`; the sums `n_decisions, n_failed, n_fallback, n_almost_optimal`; `failed_share` and `fallback_share` (each sum ÷ `n_decisions`); and `flag`, set when either share is above `threshold`. It returns the result of `sort_table`.
  - **`sort_table`** sorts by `cap` (`"on"` first) and then `eta` (ascending), directly after `allocator`.

- [ ] **Step 1: Write the failing tests**

```julia
@testset "η sweep tables" begin
    df = DataFrame(status=fill("ok", 6), allocator=["CD", "CES", "CESw", "CES", "SG", "CD"],
        ces_eta=[0.5, 0.5, 2.0, 0.1, 0.5, 0.5], w_max=[0.1, 0.1, 1.0, 0.1, 0.1, 1.0])
    e = eta_frame(df)
    @test e.allocator == ["CD", "CES", "CESw", "CES", "CD"] && e.eta == [1.0, 0.5, 2.0, 0.1, 1.0]
    @test e.cap == ["on", "on", "off", "on", "off"]
    s = DataFrame(status=fill("ok", 3), allocator=fill("CESw", 3), ces_eta=[20.0, 20.0, 2.0], w_max=fill(1.0, 3),
        n_decisions=[50, 50, 100], n_failed=[1, 1, 1], n_fallback=[0, 3, 0], n_almost_optimal=[0, 2, 0])
    t = eta_solver_table(s)
    @test t.eta == [2.0, 20.0] && t.n_runs == [1, 2] && t.failed_share ≈ [0.01, 0.02]
    @test t.fallback_share ≈ [0.0, 0.03] && t.flag == [false, true]          # 1% is not above 1%
    o = sort_table(DataFrame(allocator=fill("CES", 3), cap=["off", "on", "on"], eta=[0.1, 2.0, 0.1]))
    @test o.cap == ["on", "on", "off"] && o.eta == [0.1, 2.0, 0.1]
end
```

- [ ] **Step 2: Run it and watch it fail** (`UndefVarError: eta_frame`)

Run: `julia --project=code code/test/test_study_tables.jl`

- [ ] **Step 3: Implement the package functions**

- [ ] **Step 4: Run; it passes**

- [ ] **Step 5: Add the sweep tables to `scripts/12_tables.jl`** (not with `--smoke`)

When `eta_summary.csv` and `tier1_summary.csv` both exist, set `E = eta_frame(vcat(load_ok(tier1), load_ok(eta); cols=:union))`. Tier 1 supplies the cap-on η = 0.5 and CD points. Then append these tables:

- **`eta_sweep`**: `group_summary(E; by=[:allocator, :eta, :cap, :hedge, :screen], metrics=ETA_METRICS)`.
  Error unless every group's `n` is 30 for bandit and random, and 1 for gamma.
- **`eta_regimes`**: the same `by`, with `metrics=REGIME_METRICS`.
- **`eta_solver`**: `eta_solver_table(E)`. Print each flagged row to `stderr` as `FLAG <allocator> eta=<η> cap=<cap>: failed <share>, fallback <share>`. This is a warning, not an exit.

When `limits_summary.csv` exists, append:

- **`limits_dial`**: `eta_frame(load_ok(limits))`, selecting `allocator, eta, realized_growth, vol, max_drawdown, mean_eff_names, mean_invested, turnover, n_failed, n_fallback` and `REGIME_METRICS`.
  Error unless it has 17 rows.

The script cannot run on real sweep data until Task 9. For now, run the smoke check `julia scripts/12_tables.jl results/smoke --smoke` and expect the smoke tables to print as before. If `results/smoke` is absent, run `julia scripts/11_run_study.jl --smoke` first (2 runs).

- [ ] **Step 6: Commit**

```bash
git add code/src/StudyTables.jl code/src/Include.jl scripts/12_tables.jl code/test/test_study_tables.jl
git commit -m "feat(tables): η sweep, market-state and solver tables; the limits dial"
```

---

### Task 9: Real-data smoke and the one fresh rerun

**Files:**
- Create: `/private/tmp/claude-502/-Users-jdv27-Desktop-julia-work-INFORMS-Annual-Meeting-2026-Varnerlab/56353bd8-392f-4f30-9f95-6ae64e205c59/scratchpad/smoke_eta.jl` (scratch, not committed)
- Regenerate: `results/{tier1,fixed,universe,eta,limits}_summary.csv`, `results/limits_day.csv`, `results/tables/*.csv`

The calibration is not rerun. `results/calibration.jld2` was regenerated in 684f7e6, after the DTB3 fix, and every run's hash includes its fingerprint.

- [ ] **Step 1: Run the full suite**

Run: `julia --project=code code/test/runtests.jl`
Expected: every test passes, exit 0, more than 1,161 tests.

- [ ] **Step 2: Real-data smoke (spec §10)**

The scratch script loads `md`, `calib`, `s` and `grid` as `11_run_study.jl` does, and includes `code/test/helpers.jl` for `held_value`. It runs these into a scratch `outdir` with `run_study_grid`:
- every `limits_runs(s, grid)` run;
- the four `eta_runs` with labels `C|gamma|hedge=true|seed=5660|eta=<η>|cap=off`, for C ∈ {CES, CESw} and η ∈ {0.1, 10}.

For each result it asserts `minimum(res.cash) ≥ -1e-6` and `res.equity ≈ res.cash .+ held_value(res, md)` (rtol 1e-9), and prints `n_failed`, `n_fallback` and `n_almost_optimal`.

Expected: every assert passes and there are no errors.

- [ ] **Step 3: Run the fresh grids in the background**, in this order, logging to the scratchpad:

```
julia -t auto scripts/11_run_study.jl --fresh
julia -t auto scripts/11_run_study.jl --fixed --fresh
julia -t auto scripts/11_run_study.jl --universe --fresh
julia -t auto scripts/11_run_study.jl --eta --fresh
julia -t auto scripts/11_run_study.jl --limits --fresh
```

Expected counts and results:
- 854, 427, 4,438, 2,806 and 17 runs.
- Each prints `skipped=0 errors=0`.
- `--limits` prints its date and passes the 1e-3 gate.

If a threaded run segfaults, rerun the same command without `--fresh`; it resumes (README). Do not run `15_regression_check.jl`: this change is meant to move the results.

- [ ] **Step 4: Tables**

Run: `julia scripts/12_tables.jl`
Expected: every table is written, and the group-count and 17-row checks pass.

Copy every `FLAG` line and the `eta_solver` table into the report for the user. Points are flagged, never dropped. If the flags come from `n_almost_optimal` at every η, including CD at the headline settings, say so rather than listing each point.

- [ ] **Step 5: Commit the summaries and tables**

```bash
git add results/*_summary.csv results/limits_day.csv results/tables
git commit -m "results: fresh rerun with CESw, the η sweep and the clean limits (fractional shares, DTB3, λ)"
```

---

### Task 10: Figures

**Files:**
- Modify: `scripts/13_figures.jl`
- Regenerate: `results/figures/*.png`

Load the `dataviz` skill before writing chart code. Use the script's `style!()`, its `SLOT` colors and its 1600×900 size.

- [ ] **Step 1: Add the allocator**

Add `"CESw"` to `ALLOCATORS` after `"CES"`. Set `ALLOC_NAME["CES"] = "CES (share counts)"` and `ALLOC_NAME["CESw"] = "CES (weights)"`. Check that the existing figures still lay out with seven allocator slots.

- [ ] **Step 2: Add the new figures.** Every x axis is η on a log scale, ticked at the grid values and at 1. Every y label is a growth-rate metric in 1/yr, or a fraction or count.

- **`fig_limits_day.png`** (from `results/limits_day.csv`):
  - Two panels, CES (share counts) and CES (weights). Each name's weight is plotted against η, with Cobb-Douglas as η = 1.
  - The caption gives the date and says "budget only, L8b 13-name list, chosen in advance".
- **`fig_limits_dial.png`** (from `results/tables/limits_dial.csv`):
  - Five panels: realized growth, volatility, max drawdown, effective names and invested fraction, each against η.
  - CES and CESw lines share the CD point at η = 1.
- **`fig_eta_<metric>.png`** for each metric in `ETA_METRICS` (from `results/tables/eta_sweep.csv`):
  - Rows are cap on and cap off; columns are the screens γ, bandit and random; hedge is on.
  - CES and CESw lines show the median, with an IQR band for bandit and random. CD is marked at η = 1, and η = 0.5 is marked as the headline.
  - Caption: "Re-picks follow each run's own band exits, so the two versions can hold different baskets."
- **`fig_eta_frontier.png`**:
  - Median growth against median max drawdown, for the random screen with hedge on.
  - Two panels, cap on and cap off. Each version's points are joined in η order and labeled with η; CD is marked.

Skip each figure whose input file is missing, as figs 6–7 are skipped.

- [ ] **Step 3: Run and inspect**

Run: `julia --project=scripts scripts/13_figures.jl`
Expected: no errors. Open every new PNG and every changed one with the Read tool. Check the labels, that nothing is clipped, the log axes, and that no text says "return".

- [ ] **Step 4: Commit**

```bash
git add scripts/13_figures.jl results/figures
git commit -m "figures: the CES dial (one day, over the run) and the η sweep"
```

---

### Task 11: README and final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the README**

- **Run order:** `00 → 10 → 11 → 11 --fixed → 15 → 11 --universe → 11 --eta → 11 --limits → 12 → 13`. Give one item per new flag, with its run count and outputs (`eta_summary.csv`; `limits_summary.csv` and `limits_day.csv` with the 1e-3 gate).
- **Under "Rulings made during the build"**, one bullet each:
  - `CESw` added beside the share-count CES (user, 2026-10-21; spec `2026-10-21-ces-eta-sweep-design.md`).
  - The η sweep and the clean limits.
  - The turnover rule relabeled a spread-cost budget (user, 2026-10-22).
  - A dated note that every committed number moved in this rerun: fractional shares, DTB3, `CESw` added, and the CES cone rescaled (numerical; same argmax).
- **Under a new "CES elasticity sweep" heading in "Plan decisions"**, list:
  - the CES and CESw cone scales;
  - the scoped solver log, and the definitions of `n_fallback` and `n_almost_optimal`, including the failed hedge sub-solve;
  - the `mean_eff_names` definition;
  - the ξ_{t−1} labeling;
  - the limits day as a fresh book on day s, with the row rule;
  - the 1% flag rule (reported, never dropped);
  - the frontier figure using the random screen with hedge on;
  - the labels, and that the cap-on η = 0.5 and CD points come from Tier 1.

- [ ] **Step 2: Run the full suite one last time**

Run: `julia --project=code code/test/runtests.jl`
Expected: every test passes, exit 0. Report the count.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: the CES elasticity sweep, its run order and its decisions"
```
