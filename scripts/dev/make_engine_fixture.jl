# Engine reference case for the constrained Cobb-Douglas allocator (Task 4 oracle).
#
# Runs in the ENGINE environment, not in code/:
#   julia --project=$HOME/Desktop/julia_work/modified_cobb_douglas_example/code scripts/dev/make_engine_fixture.jl
#
# Builds one 8-asset MyConstrainedCDProblem with the engine's own build_sim_covariance (annualized
# covariance C = Δt·Σ_g), a binding σ_max = 0.12 (annualized), w_max = 0.15, nonzero n_prev, and the
# trial's turnover settings, then saves the inputs and the engine's solve_constrained_cd shares to
# code/test/fixtures/engine_cd_case.jld2. Only plain arrays and numbers are saved, so the package
# loads the fixture without the engine's types.
using ConstrainedCobbDouglas
using JLD2
using LinearAlgebra

const OUT = normpath(joinpath(@__DIR__, "..", "..", "code", "test", "fixtures", "engine_cd_case.jld2"))
const Δt = 1.0 / 252.0
const B = 100_000.0                                                   # USD
const PRICES = [45.20, 132.70, 310.40, 78.90, 512.30, 24.60, 189.00, 96.10]   # USD/share
const γ = [0.42, 0.15, -0.08, 0.31, 0.22, 0.05, -0.30, 0.18]          # two names with γ ≤ 0
const β = [0.90, 1.10, 1.30, 0.70, 1.20, 0.80, 1.00, 1.05]
const σ_ε = [3.2, 3.9, 4.6, 2.8, 4.1, 3.5, 3.0, 3.6]                  # yr⁻¹ (growth-rate units)
const σ_m = 2.6                                                       # yr⁻¹ (SPY growth SD)
const N_PREV = [210.0, 95.0, 30.0, 140.0, 22.0, 380.0, 60.0, 120.0]   # shares held before the solve
const σ_MAX = 0.12                                                    # annualized portfolio vol cap
const W_MAX = 0.15
const K_TURNOVER = 0.10 * B                                           # USD
const C̄ = 5e-4 * sum(PRICES) / length(PRICES)                         # USD/share

function main()
    ests = MySIMParameterEstimate[]
    for i in eachindex(β)
        e = MySIMParameterEstimate()
        e.ticker = "A$(i)"; e.α = 0.0; e.β = β[i]; e.σ_ε = σ_ε[i]; e.r² = 0.5
        push!(ests, e)
    end
    Σ = build_sim_covariance(ests, σ_m)                              # annualized, = Δt·Σ_g
    problem = MyConstrainedCDProblem(γ = γ, p = PRICES, B = B, Σ = Σ, σ_max = σ_MAX,
        K_turnover = K_TURNOVER, w_max = W_MAX, n_prev = N_PREV, c̄ = C̄)
    res = solve_constrained_cd(problem)
    w = res.n .* PRICES ./ B
    vol = sqrt(w' * Σ * w)
    println("status = ", res.status)
    println("n      = ", round.(res.n; digits = 4))
    println("w      = ", round.(w; digits = 5), "  Σw = ", round(sum(w); digits = 5))
    println("portfolio vol (annualized) = ", round(vol; digits = 6), "  cap = ", σ_MAX,
        "  binding = ", isapprox(vol, σ_MAX; rtol = 1e-4))
    println("max w = ", round(maximum(w); digits = 6), "  w_max = ", W_MAX)
    mkpath(dirname(OUT))
    jldsave(OUT; γ = γ, p = PRICES, B = B, Σ = Matrix{Float64}(Σ), σ_max = σ_MAX,
        K_turnover = K_TURNOVER, w_max = W_MAX, n_prev = N_PREV, c̄ = C̄, n = res.n,
        status = String(res.status), engine_vol = vol, Δt = Δt, β = β, σ_ε = σ_ε, σ_m = σ_m)
    println("wrote ", OUT)
end

main()
