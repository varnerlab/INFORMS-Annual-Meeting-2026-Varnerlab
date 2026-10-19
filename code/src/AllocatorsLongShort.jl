# 130/30 long-short Cobb-Douglas.
#
# Units as in Allocators.jl: shares n (shares; negative = short), prices S (USD/share), equity W
# (USD), weights w = n·S/W (dimensionless; negative = short). Long and short sides are financed the
# standard 130/30 way: short proceeds pay for the extra longs, net exposure Σwᵢ ≤ net_cap ≤ 1, so
# cash = 1 − Σwᵢ ≥ 0.

"""
    MyLongShortCDRule

130/30 long-short Cobb-Douglas: maximize Σ_{γᵢ>0} γᵢ·log nᵢ⁺ + Σ_{γᵢ<0} |γᵢ|·log nᵢ⁻ (shares), with
n = n⁺ − n⁻, longs only on γ > 0 names, shorts only on γ < 0 names and γ = 0 names held at zero.
Fields (dimensionless, defaults from `MyLongShortCDRule()`):
- `long_cap = 1.30`: Σ long weights ≤ long_cap
- `short_cap = 0.30`: Σ |short weights| ≤ short_cap
- `net_cap = 1.00`: Σwᵢ ≤ net_cap (must be ≤ 1: cash ≥ 0)
"""
mutable struct MyLongShortCDRule <: AbstractPortfolioRule
    long_cap::Float64
    short_cap::Float64
    net_cap::Float64
    MyLongShortCDRule() = new(1.30, 0.30, 1.00)
end

"""
    build(::Type{MyLongShortCDRule}, data::NamedTuple)::MyLongShortCDRule

Build a long-short CD rule from `MyLongShortCDRule()` (1.30 / 0.30 / 1.00), overriding any of
`long_cap`, `short_cap`, `net_cap` (dimensionless) that `data` has. Requires finite positive caps
and net_cap ≤ 1 (net exposure above 1 would need negative cash).
"""
function build(::Type{MyLongShortCDRule}, data::NamedTuple)::MyLongShortCDRule
    model = MyLongShortCDRule()
    for f ∈ (:long_cap, :short_cap, :net_cap)
        haskey(data, f) && setfield!(model, f, Float64(data[f]))
    end
    for f ∈ (:long_cap, :short_cap, :net_cap)
        v = getfield(model, f)
        (isfinite(v) && v > 0) || throw(ArgumentError("$(f) must be finite and positive, got $(v)"))
    end
    model.net_cap <= 1 || throw(ArgumentError("net_cap must be ≤ 1 (cash ≥ 0), got $(model.net_cap)"))
    return model
end

"""
    allocate(rule::MyLongShortCDRule, p::MyAllocationProblem)::MyAllocationResult

130/30 long-short CD via exponential cones on n⁺ ≥ 1e-8 (γ > 0 names) and n⁻ ≥ 1e-8 (γ < 0 names),
shares; γ = 0 names are fixed at 0. Constraints: Σ_{long} wᵢ ≤ long_cap, Σ_{short} |wᵢ| ≤ short_cap,
Σwᵢ ≤ net_cap, plus `add_shared_constraints!(…; signed=true)`: net exposure ≤ 1 (cash ≥ 0), |wᵢ| ≤
w_max, |net sector weight| ≤ s_max, √(wᵀΣ_g w) ≤ σ_g_max on the signed weights (yr⁻¹), and the
turnover budget on the signed share changes (USD). `s_star` is the net exposure Σwᵢ.

Special cases: no γ > 0 → `:no_preferred` with n = 0 (all cash, no solve); no γ < 0 →
`allocate(MyCobbDouglasRule(), p)` with status `:long_only_fallback` (a failed CD solve keeps its
status). `:infeasible` / `:solver_failed` return n = n_prev.
"""
function allocate(rule::MyLongShortCDRule, p::MyAllocationProblem)::MyAllocationResult
    _check_problem(p)
    K = length(p.γ)
    L = findall(>(0), p.γ); Sh = findall(<(0), p.γ)
    isempty(L) && return _result(p, zeros(K), :no_preferred)
    if isempty(Sh)
        r = allocate(MyCobbDouglasRule(), p)
        r.status == :optimal && (r.status = :long_only_fallback)
        return r
    end
    nL = length(L); nS = length(Sh)
    γL = p.γ[L]; γS = abs.(p.γ[Sh])
    function build_model(opt)
        m = Model(opt)
        set_silent(m)
        x⁺ = @variable(m, [1:nL], lower_bound = 1e-8)
        x⁻ = @variable(m, [1:nS], lower_bound = 1e-8)
        t⁺ = @variable(m, [1:nL])
        t⁻ = @variable(m, [1:nS])
        for k in 1:nL
            @constraint(m, [t⁺[k], 1.0, x⁺[k]] in MOI.ExponentialCone())   # t⁺ ≤ log n⁺
        end
        for k in 1:nS
            @constraint(m, [t⁻[k], 1.0, x⁻[k]] in MOI.ExponentialCone())   # t⁻ ≤ log n⁻
        end
        @objective(m, Max, sum(γL[k] * t⁺[k] for k in 1:nL) + sum(γS[k] * t⁻[k] for k in 1:nS))
        n = AffExpr[AffExpr(0.0) for _ in 1:K]                  # γ = 0 names: constant 0
        for (k, i) in enumerate(L)
            n[i] = AffExpr(0.0, x⁺[k] => 1.0)
        end
        for (k, i) in enumerate(Sh)
            n[i] = AffExpr(0.0, x⁻[k] => -1.0)
        end
        w = add_shared_constraints!(m, n, p; signed=true)
        @constraint(m, sum(w[i] for i in L) <= rule.long_cap)
        @constraint(m, sum(-w[i] for i in Sh) <= rule.short_cap)
        rule.net_cap < 1 && @constraint(m, sum(w) <= rule.net_cap)   # net ≤ 1 is the budget row
        return m, vcat(x⁺, x⁻)
    end
    x, status = _solve_with_fallback(build_model)
    status == :optimal || return _result(p, copy(p.constraints.n_prev), status)
    n = zeros(K)
    n[L] .= x[1:nL]
    n[Sh] .= -x[nL+1:end]
    return _result(p, n, :optimal)
end
