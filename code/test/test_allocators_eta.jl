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
        @test r.n[2] == 1e-3
        @test r.w ≈ budget_only_weights(rule, [0.2, -0.5, 0.3], [10.0, 20.0, 40.0]; W=1000.0) atol=1e-4
    end
end
