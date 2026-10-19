# Shared constraint set and the utility rules (Cobb-Douglas, CES, Stone–Geary).
#
# Units: shares n (shares), prices S (USD/share), equity W (USD), weights w = n·S/W (dimensionless),
# Σ_g (yr⁻²), σ_g_max (yr⁻¹). The engine's annualized cap √(wᵀCw) ≤ σ_max with C = Δt·Σ_g is the
# same constraint as √(wᵀΣ_g w) ≤ σ_g_max with σ_g_max = σ_max/√Δt.

"""
    AbstractPortfolioRule

Supertype of portfolio rules. Each rule implements `allocate(rule, p::MyAllocationProblem)`.
"""
abstract type AbstractPortfolioRule end

"""
    MyCobbDouglasRule

Cobb-Douglas (the trial's rule): maximize Σ_{γᵢ>0} γᵢ·log nᵢ; names with γᵢ ≤ 0 sit at the floor ε.
No parameters.
"""
mutable struct MyCobbDouglasRule <: AbstractPortfolioRule
    MyCobbDouglasRule() = new()
end

"""
    MyCESRule

CES utility: maximize Σ_{γᵢ>0} γᵢ·nᵢ^ρ with ρ = 1 − 1/η, a monotone transform of
(Σ γᵢ nᵢ^ρ)^(1/ρ) (CHEME-5660 `allocate_ces`: nᵢ ∝ (γᵢ/Sᵢ)^η under a budget alone).
Field `η` (elasticity of substitution, dimensionless); only η > 1 is supported. Names with γᵢ ≤ 0
sit at the floor ε.
"""
mutable struct MyCESRule <: AbstractPortfolioRule
    η::Float64
    MyCESRule() = new()
end

"""
    MyStoneGearyRule

Stone–Geary utility: maximize Σ_{γᵢ>0} γᵢ·log(nᵢ − bᵢ) with nᵢ ≥ bᵢ for every name, where the
baseline bᵢ = floor(`baseline_frac`·W/Sᵢ) shares (`baseline_frac` dimensionless, e.g. 0.01 = a 1% core
position). Names with γᵢ ≤ 0 are held at bᵢ.
"""
mutable struct MyStoneGearyRule <: AbstractPortfolioRule
    baseline_frac::Float64
    MyStoneGearyRule() = new()
end

"""
    MyConstraintSet

The constraint set shared by every portfolio rule:
- `w_max`: per-name weight cap (dimensionless; applies to |wᵢ| for signed rules)
- `s_max`: per-sector weight cap (dimensionless; `s_max ≥ 1` means no sector cap)
- `σ_g_max`: cap on the portfolio growth-rate SD √(wᵀΣ_g w) (yr⁻¹)
- `K_turnover`: turnover budget (USD), c̄·Σ|nᵢ − n_prevᵢ| ≤ K_turnover
- `c̄`: turnover cost per share (USD/share)
- `sector`: GICS sector per name
- `n_prev`: holdings before the solve (shares)
- `ε`: floor for names with γ ≤ 0 under the log/CES utilities (shares, default 1e-3)
"""
mutable struct MyConstraintSet
    w_max::Float64
    s_max::Float64
    σ_g_max::Float64
    K_turnover::Float64
    c̄::Float64
    sector::Vector{String}
    n_prev::Vector{Float64}
    ε::Float64
    MyConstraintSet() = new()
end

"""
    MyAllocationProblem

One allocation decision: `γ` (preference weights, dimensionless), `μ_g` (expected growth, yr⁻¹),
`Σ_g` (growth covariance, yr⁻²), `prices` (USD/share), `W` (equity, USD), `g_f` (risk-free growth,
yr⁻¹) and the shared `constraints::MyConstraintSet`.
"""
mutable struct MyAllocationProblem
    γ::Vector{Float64}
    μ_g::Vector{Float64}
    Σ_g::Matrix{Float64}
    prices::Vector{Float64}
    W::Float64
    g_f::Float64
    constraints::MyConstraintSet
    MyAllocationProblem() = new()
end

"""
    MyAllocationResult

Allocator output: `n` (continuous shares), `w` = n·S/W (dimensionless), `status`
(`:optimal`, `:no_preferred`, `:infeasible`, `:solver_failed`, plus rule-specific statuses), and
`s_star` (invested fraction of equity, Σwᵢ for the utility rules).
"""
mutable struct MyAllocationResult
    n::Vector{Float64}
    w::Vector{Float64}
    status::Symbol
    s_star::Float64
    MyAllocationResult() = new()
end

_require(data::NamedTuple, T::Type, f::Symbol) =
    haskey(data, f) || throw(ArgumentError("Cannot build $(T): missing field $(f)"))

"""
    build(::Type{MyCobbDouglasRule}, data::NamedTuple)::MyCobbDouglasRule

Build a Cobb-Douglas rule (no fields; `data` is ignored).
"""
build(::Type{MyCobbDouglasRule}, data::NamedTuple)::MyCobbDouglasRule = MyCobbDouglasRule()

"""
    build(::Type{MyCESRule}, data::NamedTuple)::MyCESRule

Build a CES rule from `(η = …,)` (elasticity, dimensionless, finite and > 0; `allocate` needs η > 1).
"""
function build(::Type{MyCESRule}, data::NamedTuple)::MyCESRule
    _require(data, MyCESRule, :η)
    η = Float64(data.η)
    (isfinite(η) && η > 0) || throw(ArgumentError("η must be finite and positive, got $(η)"))
    model = MyCESRule(); model.η = η
    return model
end

"""
    build(::Type{MyStoneGearyRule}, data::NamedTuple)::MyStoneGearyRule

Build a Stone–Geary rule from `(baseline_frac = …,)` (dollar baseline per name as a fraction of
equity, dimensionless, ≥ 0).
"""
function build(::Type{MyStoneGearyRule}, data::NamedTuple)::MyStoneGearyRule
    _require(data, MyStoneGearyRule, :baseline_frac)
    f = Float64(data.baseline_frac)
    (isfinite(f) && f >= 0) || throw(ArgumentError("baseline_frac must be finite and ≥ 0, got $(f)"))
    model = MyStoneGearyRule(); model.baseline_frac = f
    return model
end

"""
    build(::Type{MyConstraintSet}, data::NamedTuple)::MyConstraintSet

Build the shared constraint set from `w_max`, `s_max`, `σ_g_max` (yr⁻¹), `K_turnover` (USD),
`c̄` (USD/share), `sector`, `n_prev` (shares) and optional `ε` (shares, default 1e-3).
"""
function build(::Type{MyConstraintSet}, data::NamedTuple)::MyConstraintSet
    for f ∈ (:w_max, :s_max, :σ_g_max, :K_turnover, :c̄, :sector, :n_prev)
        _require(data, MyConstraintSet, f)
    end
    model = MyConstraintSet()
    model.w_max = Float64(data.w_max)
    model.s_max = Float64(data.s_max)
    model.σ_g_max = Float64(data.σ_g_max)
    model.K_turnover = Float64(data.K_turnover)
    model.c̄ = Float64(data.c̄)
    model.sector = String.(collect(data.sector))
    model.n_prev = Float64.(collect(data.n_prev))
    model.ε = Float64(get(data, :ε, 1e-3))
    model.w_max > 0 || throw(ArgumentError("w_max must be positive"))
    model.s_max > 0 || throw(ArgumentError("s_max must be positive"))
    model.σ_g_max >= 0 || throw(ArgumentError("σ_g_max must be ≥ 0"))
    (model.K_turnover >= 0 && model.c̄ >= 0) || throw(ArgumentError("K_turnover and c̄ must be ≥ 0"))
    model.ε >= 0 || throw(ArgumentError("ε must be ≥ 0"))
    length(model.sector) == length(model.n_prev) ||
        throw(DimensionMismatch("sector and n_prev must have one entry per name"))
    return model
end

"""
    build(::Type{MyAllocationProblem}, data::NamedTuple)::MyAllocationProblem

Build an allocation problem from `γ`, `μ_g` (yr⁻¹), `Σ_g` (yr⁻², any K×K matrix type), `prices`
(USD/share), `W` (USD), `g_f` (yr⁻¹) and `constraints::MyConstraintSet`. Checks that every per-name
input has the same length K and that prices and W are positive.
"""
function build(::Type{MyAllocationProblem}, data::NamedTuple)::MyAllocationProblem
    for f ∈ (:γ, :μ_g, :Σ_g, :prices, :W, :g_f, :constraints)
        _require(data, MyAllocationProblem, f)
    end
    model = MyAllocationProblem()
    model.γ = Float64.(collect(data.γ))
    model.μ_g = Float64.(collect(data.μ_g))
    model.Σ_g = Matrix{Float64}(data.Σ_g)
    model.prices = Float64.(collect(data.prices))
    model.W = Float64(data.W)
    model.g_f = Float64(data.g_f)
    model.constraints = data.constraints
    _check_problem(model)
    return model
end

"""
    build(::Type{MyAllocationResult}, data::NamedTuple)::MyAllocationResult

Build a result from `n` (shares), `w` (dimensionless), `status::Symbol` and `s_star` (dimensionless).
"""
function build(::Type{MyAllocationResult}, data::NamedTuple)::MyAllocationResult
    for f ∈ (:n, :w, :status, :s_star)
        _require(data, MyAllocationResult, f)
    end
    model = MyAllocationResult()
    model.n = Float64.(collect(data.n)); model.w = Float64.(collect(data.w))
    model.status = Symbol(data.status); model.s_star = Float64(data.s_star)
    return model
end

# Every per-name input has length K; Σ_g is K×K; prices and W are positive and finite.
function _check_problem(p::MyAllocationProblem)
    K = length(p.γ)
    c = p.constraints
    all(length(v) == K for v in (p.μ_g, p.prices, c.sector, c.n_prev)) ||
        throw(DimensionMismatch("γ, μ_g, prices, sector and n_prev must all have length $(K)"))
    size(p.Σ_g) == (K, K) || throw(DimensionMismatch("Σ_g must be $(K)×$(K), got $(size(p.Σ_g))"))
    all(s -> isfinite(s) && s > 0, p.prices) || throw(ArgumentError("prices must be positive and finite"))
    (isfinite(p.W) && p.W > 0) || throw(ArgumentError("W must be positive and finite"))
    return nothing
end

# Result with w = n·S/W and s_star = Σw (the invested fraction of equity).
function _result(p::MyAllocationProblem, n::AbstractVector, status::Symbol)::MyAllocationResult
    w = n .* p.prices ./ p.W
    return build(MyAllocationResult, (n=n, w=w, status=status, s_star=sum(w)))
end

# A pinned entry of the share vector is a constant (no decision-variable terms).
_is_pinned(x::Real) = true
_is_pinned(x::JuMP.GenericAffExpr) = isempty(x.terms)
_is_pinned(x::JuMP.AbstractVariableRef) = false
_pinned_value(x::Real) = Float64(x)
_pinned_value(x::JuMP.GenericAffExpr) = Float64(x.constant)

# lhs ≤ rhs. A constant row (pinned names only) is checked here instead of being sent to the
# solver, because all-zero rows degrade Clarabel's accuracy (CD with one ε pin: share error 1.3e-5
# with them, 2e-6 without). A violated constant row is still added, so the solver certifies the
# infeasibility.
function _add_le!(m::JuMP.Model, lhs::JuMP.GenericAffExpr, rhs)
    isempty(lhs.terms) && rhs isa Real && lhs.constant <= rhs && return nothing
    @constraint(m, lhs <= rhs)
    return nothing
end

"""
    add_shared_constraints!(m::JuMP.Model, n, p::MyAllocationProblem; signed=false, κ=1.0)

Add the shared constraint set to `m` for the share vector `n` (length K, shares; entries are JuMP
variables/affine expressions, or constants for pinned names) and return the weight expressions
w = n·S/W (dimensionless, `Vector{AffExpr}`). Constraints, following engine `Allocator.jl:115-163`:

- **Budget:** Σ nᵢSᵢ ≤ W·κ (USD). With `signed=true` this is net exposure ≤ 1 (cash ≥ 0).
- **Per-name cap:** wᵢ ≤ w_max·κ for every name, pinned ones included (a pinned name's constant row
  is checked directly and only sent to the solver when violated); with `signed=true` also
  wᵢ ≥ −w_max·κ, i.e. |wᵢ| ≤ w_max·κ.
- **Sector cap** (only when `s_max < 1`): Σ_{i∈s} wᵢ ≤ s_max·κ per sector; with `signed=true` the
  net sector weight is capped on both sides, |Σ_{i∈s} wᵢ| ≤ s_max·κ.
- **Volatility cap:** ‖Lᵀw‖ ≤ σ_g_max·κ (yr⁻¹) as a second-order cone, LLᵀ = Σ_g + 1e-8·I (yr⁻²);
  skipped when σ_g_max is infinite.
- **Turnover budget:** c̄·Σ_{free i} |nᵢ − κ·n_prevᵢ| ≤ K_turnover·κ (USD), on the non-pinned names
  only (engine semantics: pinned names' churn is not counted). The slack variables are skipped when
  c̄ = 0 or when no feasible point can violate the budget: c̄·max‖n − n_prev‖₁ ≤ K_turnover, with
  max‖·‖₁ ≤ B_eff/min Sᵢ + Σ n_prevᵢ (long-only; B_eff = W − pinned cost, USD) or
  Σ (w_max·W/Sᵢ + |n_prevᵢ|) (signed), both over the free names.

The sign of `n` is the caller's: long-only rules declare nᵢ ≥ 0 (or a floor) as variable bounds,
and the long-only turnover guard relies on it; signed rules pass n = n⁺ − n⁻.
`κ` (dimensionless, a number or a JuMP variable) homogenizes every right-hand side for
fractional programs (y = κ·n, e.g. the tangent rule); with a variable κ, pass no constant pins.
"""
function add_shared_constraints!(m::JuMP.Model, n::AbstractVector, p::MyAllocationProblem;
        signed::Bool=false, κ::Union{Real,JuMP.AbstractVariableRef}=1.0)
    c = p.constraints; S = p.prices; W = p.W; K = length(S)
    length(n) == K || throw(DimensionMismatch("n must have length $(K)"))

    # Weights as AffExpr: pinned entries are constants, so build the vector explicitly to keep a
    # concrete AffExpr element type (the engine's SOC typing workaround).
    w = AffExpr[AffExpr(0.0) for _ in 1:K]
    for i in 1:K
        add_to_expression!(w[i], S[i] / W, n[i])
    end

    # Budget (USD)
    budget = AffExpr(0.0)
    for i in 1:K
        add_to_expression!(budget, S[i], n[i])
    end
    @constraint(m, budget <= W * κ)

    # Per-name cap
    for i in 1:K
        _add_le!(m, w[i], c.w_max * κ)
        signed && _add_le!(m, -w[i], c.w_max * κ)
    end

    # Sector cap (net sector weight)
    if c.s_max < 1.0
        for s in unique(c.sector)
            ws = sum(w[i] for i in 1:K if c.sector[i] == s)
            _add_le!(m, ws, c.s_max * κ)
            signed && _add_le!(m, -ws, c.s_max * κ)
        end
    end

    # Volatility cap: ‖Lᵀw‖ ≤ σ_g_max
    if isfinite(c.σ_g_max)
        Lt = Matrix(cholesky(Symmetric(p.Σ_g + 1e-8 * I)).L')
        Ltw = AffExpr[AffExpr(0.0) for _ in 1:K]
        for r in 1:K, col in r:K                     # Lᵀ is upper triangular
            add_to_expression!(Ltw[r], Lt[r, col], w[col])
        end
        @constraint(m, vcat(c.σ_g_max * κ, Ltw) in SecondOrderCone())
    end

    # Turnover budget (ℓ₁ slack on the free names), skipped when it cannot bind
    if c.c̄ > 0
        free = [i for i in 1:K if !_is_pinned(n[i])]
        if !isempty(free)
            max_l1 = if signed
                sum(c.w_max * W / S[i] + abs(c.n_prev[i]) for i in free)
            else
                B_eff = W - sum((_pinned_value(n[i]) * S[i] for i in 1:K if _is_pinned(n[i])); init=0.0)
                B_eff / minimum(S[free]) + sum(c.n_prev[free])
            end
            if c.c̄ * max_l1 > c.K_turnover
                u = @variable(m, [1:length(free)], lower_bound = 0.0)
                for (k, i) in enumerate(free)
                    @constraint(m, u[k] >= n[i] - κ * c.n_prev[i])
                    @constraint(m, u[k] >= κ * c.n_prev[i] - n[i])
                end
                @constraint(m, c.c̄ * sum(u) <= c.K_turnover * κ)
            end
        end
    end
    return w
end

# Gap tolerances are tighter than the engine's 1e-9: the utility is flat at the optimum, so share
# accuracy scales like √gap (3-asset CD closed form off by 2.5e-5 at 1e-9, by 6e-9 at 1e-12). The
# feasibility tolerance stays at the engine's 1e-9.
const _CLARABEL_ATTRIBUTES = ("tol_gap_abs" => 1e-12, "tol_gap_rel" => 1e-12, "tol_feas" => 1e-9,
    "max_iter" => 500)
const _ACCEPTED = (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
const _INFEASIBLE = (MOI.INFEASIBLE, MOI.ALMOST_INFEASIBLE, MOI.LOCALLY_INFEASIBLE,
    MOI.INFEASIBLE_OR_UNBOUNDED)

# Solve with Clarabel (tight tolerances), then SCS if Clarabel does not reach (almost) optimal.
# `build_model(optimizer)` returns `(model, x)`. Returns `(x_value or nothing, status)` with status
# :optimal, :infeasible (either solver certified infeasibility) or :solver_failed.
function _solve_with_fallback(build_model::Function)
    statuses = MOI.TerminationStatusCode[]
    for opt in (optimizer_with_attributes(Clarabel.Optimizer, _CLARABEL_ATTRIBUTES...), SCS.Optimizer)
        m, x = build_model(opt)
        optimize!(m)
        st = termination_status(m)
        push!(statuses, st)
        st in _ACCEPTED && return (value.(x), :optimal)
    end
    return (nothing, any(in(_INFEASIBLE), statuses) ? :infeasible : :solver_failed)
end

# Utility solve over the preferred names (γ > 0): `pins` gives every name's value when pinned,
# `lb` the lower bound (shares) of each preferred name's variable, and `objective!(m, x, γ_p)` adds
# the cone constraints and the objective. Pinned names enter the shared constraints as constants.
function _solve_utility(p::MyAllocationProblem, pins::Vector{Float64}, lb::Vector{Float64},
        objective!::Function)::MyAllocationResult
    pref = findall(p.γ .> 0)
    K = length(p.γ); Kp = length(pref)
    function build_model(opt)
        m = Model(opt)
        set_silent(m)
        x = @variable(m, [k = 1:Kp], lower_bound = lb[k])
        n = AffExpr[AffExpr(pins[i]) for i in 1:K]
        for (k, i) in enumerate(pref)
            n[i] = AffExpr(0.0, x[k] => 1.0)
        end
        objective!(m, x, p.γ[pref])
        add_shared_constraints!(m, n, p)
        return m, x
    end
    x, status = _solve_with_fallback(build_model)
    status == :optimal || return _result(p, copy(p.constraints.n_prev), status)
    n = copy(pins)
    n[pref] .= x
    return _result(p, n, :optimal)
end

"""
    allocate(rule::AbstractPortfolioRule, p::MyAllocationProblem)::MyAllocationResult

Solve one allocation under the shared constraints (`add_shared_constraints!`). Shares `n` are
continuous (shares), weights w = n·S/W. Status `:no_preferred` when no γᵢ > 0 (utility rules return
their pins, no solve); `:infeasible` or `:solver_failed` return `n = constraints.n_prev`.
"""
function allocate end

"""
    allocate(::MyCobbDouglasRule, p::MyAllocationProblem)::MyAllocationResult

Cobb-Douglas: maximize Σ_{γ>0} γᵢ·log nᵢ (nᵢ in shares) via exponential cones, with nᵢ ≥ 1e-8 and
γ ≤ 0 names pinned at ε shares.
"""
function allocate(::MyCobbDouglasRule, p::MyAllocationProblem)::MyAllocationResult
    _check_problem(p)
    pins = fill(p.constraints.ε, length(p.γ))
    any(>(0), p.γ) || return _result(p, pins, :no_preferred)
    Kp = count(>(0), p.γ)
    return _solve_utility(p, pins, fill(1e-8, Kp), function (m, x, γp)
        t = @variable(m, [1:Kp])
        for k in 1:Kp
            @constraint(m, [t[k], 1.0, x[k]] in MOI.ExponentialCone())   # t ≤ log n
        end
        @objective(m, Max, sum(γp[k] * t[k] for k in 1:Kp))
    end)
end

"""
    allocate(rule::MyCESRule, p::MyAllocationProblem)::MyAllocationResult

CES: maximize Σ_{γ>0} γᵢ·tᵢ with (nᵢ, 1, tᵢ) ∈ PowerCone(ρ), ρ = 1 − 1/η, i.e. tᵢ ≤ nᵢ^ρ (nᵢ in
shares); γ ≤ 0 names pinned at ε shares. Throws `ArgumentError` for η ≤ 1.
"""
function allocate(rule::MyCESRule, p::MyAllocationProblem)::MyAllocationResult
    rule.η > 1 || throw(ArgumentError("η ≤ 1 not supported yet"))
    _check_problem(p)
    ρ = 1 - 1 / rule.η
    pins = fill(p.constraints.ε, length(p.γ))
    any(>(0), p.γ) || return _result(p, pins, :no_preferred)
    Kp = count(>(0), p.γ)
    return _solve_utility(p, pins, zeros(Kp), function (m, x, γp)
        t = @variable(m, [1:Kp])
        for k in 1:Kp
            @constraint(m, [x[k], 1.0, t[k]] in MOI.PowerCone(ρ))         # |t| ≤ n^ρ
        end
        @objective(m, Max, sum(γp[k] * t[k] for k in 1:Kp))
    end)
end

"""
    allocate(rule::MyStoneGearyRule, p::MyAllocationProblem)::MyAllocationResult

Stone–Geary: maximize Σ_{γ>0} γᵢ·log(nᵢ − bᵢ) via exponential cones on the excess shares, with
bᵢ = floor(baseline_frac·W/Sᵢ) shares, nᵢ ≥ bᵢ, and γ ≤ 0 names held at bᵢ. Returns `:infeasible`
with n_prev when Σ bᵢSᵢ > W (USD); `:no_preferred` returns n = b.
"""
function allocate(rule::MyStoneGearyRule, p::MyAllocationProblem)::MyAllocationResult
    _check_problem(p)
    b = floor.(rule.baseline_frac * p.W ./ p.prices)                      # baseline shares
    sum(b .* p.prices) > p.W && return _result(p, copy(p.constraints.n_prev), :infeasible)
    any(>(0), p.γ) || return _result(p, b, :no_preferred)
    pref = findall(p.γ .> 0); Kp = length(pref)
    b_p = b[pref]
    return _solve_utility(p, b, b_p, function (m, x, γp)
        t = @variable(m, [1:Kp])
        for k in 1:Kp
            @constraint(m, [t[k], 1.0, x[k] - b_p[k]] in MOI.ExponentialCone())  # t ≤ log(n − b)
        end
        @objective(m, Max, sum(γp[k] * t[k] for k in 1:Kp))
    end)
end
