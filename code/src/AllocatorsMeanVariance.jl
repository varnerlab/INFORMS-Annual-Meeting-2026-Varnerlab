# Min-variance and tangent (maximum growth-Sharpe) rules, the investable fraction s*, and the
# expected-growth vector every rule feeds into its problem.
#
# Units as in Allocators.jl: shares n (shares), prices S (USD/share), equity W (USD), weights
# w = n·S/W (dimensionless), μ_g and g_f (yr⁻¹), Σ_g (yr⁻²). Neither rule uses γ: every name is free
# (n ≥ 0, no ε pins), and both invest s* = the largest fraction of equity the shared constraints allow,
# up to a tolerance of `_S_TOL` (ruling R22).

"""
    MyMinVarRule

Minimum variance: minimize wᵀΣ_g w (yr⁻²) subject to the shared constraints and Σwᵢ = s*
(`max_investable`). No parameters; γ plays no role.
"""
mutable struct MyMinVarRule <: AbstractPortfolioRule
    MyMinVarRule() = new()
end

"""
    MyTangentRule

Tangent (maximum growth-Sharpe): maximize (μ_g − g_f)ᵀw / √(wᵀΣ_g w) (dimensionless) subject to the
shared constraints and Σwᵢ = s*. Field `zero_alpha::Bool` (default `false`): when `true`,
`expected_growth` drops α and uses μ_g = β·ḡ_M. γ plays no role.
"""
mutable struct MyTangentRule <: AbstractPortfolioRule
    zero_alpha::Bool
    MyTangentRule() = new(false)
end

"""
    build(::Type{MyMinVarRule}, data::NamedTuple)::MyMinVarRule

Build a min-variance rule (no fields; `data` is ignored).
"""
build(::Type{MyMinVarRule}, data::NamedTuple)::MyMinVarRule = MyMinVarRule()

"""
    build(::Type{MyTangentRule}, data::NamedTuple)::MyTangentRule

Build a tangent rule from `MyTangentRule()` (zero_alpha = false), overriding `zero_alpha::Bool` when
`data` has it.
"""
function build(::Type{MyTangentRule}, data::NamedTuple)::MyTangentRule
    model = MyTangentRule()
    if haskey(data, :zero_alpha)
        data.zero_alpha isa Bool ||
            throw(ArgumentError("zero_alpha must be a Bool, got $(repr(data.zero_alpha))"))
        model.zero_alpha = data.zero_alpha
    end
    return model
end

"""
    expected_growth(rule::AbstractPortfolioRule, α::AbstractVector, β::AbstractVector, ḡ_M::Real)::Vector{Float64}

Expected growth μ_g,i = αᵢ + βᵢ·ḡ_M (yr⁻¹), with α the SIM intercepts (yr⁻¹), β the SIM betas
(dimensionless) and ḡ_M the trailing 252-day mean SPY growth rate (yr⁻¹). `MyTangentRule` with
`zero_alpha = true` returns βᵢ·ḡ_M.
"""
function expected_growth(rule::AbstractPortfolioRule, α::AbstractVector, β::AbstractVector,
        ḡ_M::Real)::Vector{Float64}
    _check_sim_lengths(α, β)
    return Float64.(α .+ β .* ḡ_M)
end

function expected_growth(rule::MyTangentRule, α::AbstractVector, β::AbstractVector,
        ḡ_M::Real)::Vector{Float64}
    _check_sim_lengths(α, β)
    return rule.zero_alpha ? Float64.(β .* ḡ_M) : Float64.(α .+ β .* ḡ_M)
end

_check_sim_lengths(α, β) = length(α) == length(β) ||
    throw(DimensionMismatch("α and β must have the same length, got $(length(α)) and $(length(β))"))

# Tolerance on Σw = s* (dimensionless): the rules impose s* − _S_TOL ≤ Σw, and the shared
# constraints already cap Σw at s*, so the solver never meets a marginally infeasible equality (R22).
const _S_TOL = 1e-7

# Below this fraction of the largest attainable excess growth s*·max|eᵢ| (yr⁻¹), the best feasible
# excess growth counts as zero (solver tolerance): the tangent rule then falls back to min-variance.
const _MIN_EXCESS_REL = 1e-6

# Model over free shares n ≥ 0 (no pins) under the shared constraints; returns (m, n, w). With `s`
# given, also Σw ≥ s − _S_TOL.
function _free_model(opt, p::MyAllocationProblem; s::Union{Nothing,Float64}=nothing)
    m = Model(opt)
    set_silent(m)
    n = @variable(m, [1:length(p.γ)], lower_bound = 0.0)
    w = add_shared_constraints!(m, n, p)
    s === nothing || @constraint(m, sum(w) >= s - _S_TOL)
    return m, n, w
end

# s* and the solve status (:optimal, or :infeasible / :solver_failed with s* = NaN).
function _max_investable(p::MyAllocationProblem)::Tuple{Float64,Symbol}
    x, status = _solve_with_fallback(function (opt)
        m, n, w = _free_model(opt, p)
        @objective(m, Max, sum(w))
        return m, n
    end)
    status == :optimal || return (NaN, status)
    return (clamp(sum(x .* p.prices) / p.W, 0.0, 1.0), :optimal)
end

"""
    max_investable(p::MyAllocationProblem)::Float64

Investable fraction s* = min(1, max Σwᵢ) (dimensionless) over long-only shares n ≥ 0 (no ε pins)
under the shared constraints (`add_shared_constraints!`): the budget, per-name, sector, volatility
and turnover limits. Returns `NaN` when the shared constraints have no feasible point (or both
solvers fail); the rules then report `:infeasible` / `:solver_failed` and keep the previous holdings.
"""
function max_investable(p::MyAllocationProblem)::Float64
    _check_problem(p)
    return first(_max_investable(p))
end

# Result whose s_star is the target s* rather than Σw.
function _result_at(p::MyAllocationProblem, n::AbstractVector, status::Symbol, s::Float64)
    r = _result(p, n, status)
    r.s_star = s
    return r
end

# Min-variance at a given s*: minimize wᵀΣ_g w with s* − _S_TOL ≤ Σw.
function _minvar(p::MyAllocationProblem, s::Float64)::MyAllocationResult
    x, status = _solve_with_fallback(function (opt)
        m, n, w = _free_model(opt, p; s=s)
        @objective(m, Min, w' * p.Σ_g * w)
        return m, n
    end)
    status == :optimal || return _result(p, copy(p.constraints.n_prev), status)
    return _result_at(p, x, :optimal, s)
end

"""
    allocate(::MyMinVarRule, p::MyAllocationProblem)::MyAllocationResult

Minimum variance: minimize wᵀΣ_g w (yr⁻²) over long-only shares n ≥ 0 (shares, no ε pins) under the
shared constraints, with Σwᵢ = s* = `max_investable(p)` (to within 1e-7). `s_star` = s*. Returns
`:infeasible` / `:solver_failed` with n = n_prev when the shared constraints cannot be met.
"""
function allocate(::MyMinVarRule, p::MyAllocationProblem)::MyAllocationResult
    _check_problem(p)
    s, status = _max_investable(p)
    status == :optimal || return _result(p, copy(p.constraints.n_prev), status)
    return _minvar(p, s)
end

# Min-variance relabelled :fallback_minvar (a failed min-variance solve keeps its status).
function _fallback_minvar(p::MyAllocationProblem, s::Float64)::MyAllocationResult
    r = _minvar(p, s)
    r.status == :optimal && (r.status = :fallback_minvar)
    return r
end

"""
    allocate(::MyTangentRule, p::MyAllocationProblem)::MyAllocationResult

Tangent portfolio: maximize eᵀw / √(wᵀΣ_g w) with excess growth e = μ_g − g_f (yr⁻¹), over long-only
shares (no ε pins) under the shared constraints, with Σwᵢ = s* (to within 1e-7). `s_star` = s*.

Solved in two steps:
1. **Best excess growth** δ = max eᵀw (yr⁻¹) over the same feasible set (a cone LP). When δ is not
   positive (δ ≤ 1e-6·s*·max|eᵢ|, the solver tolerance), no feasible portfolio has positive excess
   growth and the rule returns the min-variance solution with status `:fallback_minvar`.
2. **Charnes–Cooper homogenization.** With y = κ·n (shares) and κ ≥ 0, and v = κ·w the weights
   `add_shared_constraints!(m, y, p; κ=κ)` returns: minimize vᵀΣ_g v subject to eᵀv = δ,
   Σv ≥ (s* − 1e-7)·κ, y ≥ 0, and every shared constraint with its right-hand side scaled by κ
   (budget, caps, sector, volatility SOC and turnover all stay linear or SOC in (y, κ)). Then
   w = v/κ and n = y/κ. Fixing eᵀv to δ instead of 1 leaves the optimum unchanged (the Sharpe ratio
   is scale-free) and keeps κ = δ/eᵀw ≥ 1, so dividing by κ never amplifies solver error.

Returns `:infeasible` / `:solver_failed` with n = n_prev when a solve fails.
"""
function allocate(::MyTangentRule, p::MyAllocationProblem)::MyAllocationResult
    _check_problem(p)
    K = length(p.γ)
    n_prev = p.constraints.n_prev
    s, status = _max_investable(p)
    status == :optimal || return _result(p, copy(n_prev), status)
    e = p.μ_g .- p.g_f

    # 1. best feasible excess growth δ
    x, status = _solve_with_fallback(function (opt)
        m, n, w = _free_model(opt, p; s=s)
        @objective(m, Max, sum(e[i] * w[i] for i in 1:K))
        return m, n
    end)
    status == :optimal || return _result(p, copy(n_prev), status)
    δ = dot(e, x .* p.prices ./ p.W)
    scale = s * maximum(abs, e)
    (scale > 0 && δ > _MIN_EXCESS_REL * scale) || return _fallback_minvar(p, s)

    # 2. homogenized maximum-Sharpe problem in (y, κ)
    x, status = _solve_with_fallback(function (opt)
        m = Model(opt)
        set_silent(m)
        y = @variable(m, [1:K], lower_bound = 0.0)
        κ = @variable(m, lower_bound = 0.0)
        v = add_shared_constraints!(m, y, p; κ=κ)                # v = κ·w
        @constraint(m, sum(v) >= (s - _S_TOL) * κ)
        @constraint(m, sum(e[i] * v[i] for i in 1:K) == δ)
        @objective(m, Min, v' * p.Σ_g * v)
        return m, vcat(y, κ)
    end)
    status == :infeasible && return _fallback_minvar(p, s)      # not expected once δ > 0
    status == :optimal || return _result(p, copy(n_prev), status)
    return _result_at(p, x[1:K] ./ x[end], :optimal, s)
end
